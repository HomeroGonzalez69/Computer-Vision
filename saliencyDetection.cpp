/*#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>
//import kagglehub

//path = kagglehub.dataset_download("balraj98/duts-saliency-detection-dataset")

//print("Path to dataset files:", path)

using namespace std;
using namespace cv;


// Edge node used by D's algo
struct Edge {
    int node;
    double weight;
};

//Priority Queue entry
struct QueueNode{
    int node;
    double distance;

    bool operator>(const QueueNode& other) const{
        return distance > other.distance;
    }
};

// Converts image coordinates to graph vertex index
int pixelToNode(int x, int y, int width){
    return y * width + x;
}

// Converts graph vertex index back to image coords
Point nodeToPixel (int node, int width){
    return Point(node % width, node / width);
}

// Calculates a contrast based saliency map
Mat calculateSaliency(const Mat& image){
    Mat blurred;
    Mat difference;
    Mat floatImage;

    //converts image to floating point before any multiplication calculations
    image.convertTo(floatImage, CV_64FC3, 1.0 /255.0);


    // Blur the image to assist in calculations
    GaussianBlur(floatImage, blurred, Size(0,0),9.0);

    // Calculate difference between original and blurred image
    absdiff(floatImage, blurred, difference);

    // Magnitude of color difference
    vector<Mat> channels;
    split(difference, channels);

    Mat magnitudeSquared = channels[0].mul(channels[0]) +
        channels[1].mul(channels[1]) + channels[2].mul(channels[2]);
    
    //calculates Eucladian magnitude
    Mat saliency;
    sqrt(magnitudeSquared, saliency);

    // Apply normalization to saliency
    normalize(saliency, saliency, 0.0, 1.0, NORM_MINMAX);

    //convert to VC_64F so the program can still access saliency at <double>
    saliency.convertTo(saliency, CV_64F);

    return saliency;
}

// Calculates normalized color difference between two pixels
double colorDifference(const Vec3b& a, const Vec3b& b){
    double db = static_cast<double>(a[0]) - b[0];
    double dg = static_cast<double>(a[1]) - b[1];
    double dr = static_cast<double>(a[2]) - b[2];

    //Pythagorean theorum
    double distance = sqrt(db * db + dg * dg + dr * dr);

    // Maximum Eucladian distance for two pixels is sqrt(3 * 255^2)
    return distance / (255.0 * sqrt(3.0));
}

// calculates the cost of moving between pixels
double calculateEdgeWeight(const Mat& image, const Mat& saliency, int x1, int y1, int x2, int y2){
    double destinationSaliency = saliency.at<double>(y2, x2);
    double colorDistance = colorDifference(image.at<Vec3b>(y1, x1), image.at<Vec3b>(y2, x2));
    
    // assigns parameters of formula
    const double epsilon = 0.01; // makes sure the fucntion is positive
    const double alpha = .5; // higher means more restrictive on saliency edge cost
    const double beta = 2.0; // Controls influence of color differences makes it more expensive for D's to cross the strong color boundries

    // computes edge weights
    return epsilon + alpha * (1.0 - destinationSaliency) + beta * colorDistance;
}

// Creates a 4 way connected graph from the image
vector<vector<Edge>> buildGraph(const Mat& image, const Mat& saliency){
    int width = image.cols;
    int height = image.rows;
    int totalNodes = width * height;

    vector<vector<Edge>> graph(totalNodes);

    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};

    for (int y = 0; y < height; ++y){
        for(int x = 0; x< width; ++x){
            int currentNode = pixelToNode(x, y, width);

            for (int direction = 0; direction < 4; ++direction){
                int nx = x + dx[direction];
                int ny = y + dy[direction];

                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                
                int neightborNode = pixelToNode(nx, ny, width);

                double weight = calculateEdgeWeight(image, saliency, x, y, nx, ny);

                graph[currentNode].push_back({neightborNode, weight});
            }
        }
    }
    return graph;
}

// Using D's from starting pixel
vector<double> dijkstra(const vector<vector<Edge>>& graph, int source){
    int totalNodes = static_cast<int>(graph.size());

    vector<double> distances(totalNodes, numeric_limits<double>::infinity());

    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> pq;

    distances[source] = 0.0;
    pq.push({source, 0.0});

    while(!pq.empty()){
        QueueNode current = pq.top();
        pq.pop();

        //ignores outdated entries
        if (current.distance > distances[current.node]){
            continue;
        }
        for (const Edge& edge : graph[current.node]){
            double newDistance = current.distance + edge.weight;

            if (newDistance < distances[edge.node]){
                distances[edge.node] = newDistance;

                pq.push({
                    edge.node, newDistance
                });
            }
        }
    }
    return distances;
}

// Multi source Dijkstra's shortest path algorithm
vector<double> dijkstra(const vector<vector<Edge>>& graph, const vector<int>& sources){
    int totalNodes = static_cast<int>(graph.size());

    // Initializes all distances to infinity
    vector<double> distances(totalNodes, numeric_limits<double>::infinity());

    // Priority queue for the shortest path exploration
    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> pq;

    // Initializes every source with distance zero
    for (int source : sources){
        if (source < 0 || source >= totalNodes){
            throw out_of_range("Invalid Dijkstra source node.");
        }
        // Instead of using one vertex every selected source vertex is used, priority queue then handles them all together
        distances[source] = 0.0;
        pq.push({source, 0.0});
    }

    // runs Dijsktra's algorithm
    while(!pq.empty()){
        QueueNode current = pq.top();
        pq.pop();

        // Ignores outdated queue entries
        if (current.distance > distances[current.node]){
            continue;
        }
        //explores neighboring verticies
        for (const Edge& edge : graph[current.node]){
            double newDistance = current.distance + edge.weight;
            if(newDistance < distances[edge.node]){
                distances[edge.node] = newDistance;

                pq.push({edge.node, newDistance});
            }
        }
    }
    return distances;
}

// Calculates combined saliency using original saliency
// and Dijkstra shortest-path distances
// tau controls how quickly the influence of the score decreases, smaller values create more localized regions
// Larger values allow the sources influence to extend further, very large values make original saliency map dominate the results
Mat calculateCombinedSaliency(const Mat& saliency, const vector<double>& distances, double tau){
    Mat combinedSaliency(saliency.rows, saliency.cols, CV_64F, Scalar(0));

    int width = saliency.cols;

    for (int y=0; y < saliency.rows; ++y){
        for (int x=0; x<saliency.cols;++x){
            int node = pixelToNode(x, y, width);

            double originalSaliency = saliency.at<double>(y, x);

            double pathDistance = distances[node];

            // ignore unreachable pixels
            if (!isfinite(pathDistance)){
                continue;
            }

            // calculate distance-based influence
            double pathInfluence = exp(-pathDistance / tau);

            // Combine original saliency with path influence 
            double combinedScore = originalSaliency * pathInfluence;

            combinedSaliency.at<double>(y, x) = combinedScore;
        }
    }
    // returns new matrix containing combined saliency scores
    return combinedSaliency;
}

// Finds the pixel with the highest saliency
Point findMostSalientPixel(const Mat& saliency){
    Point maximumLocation;

    minMaxLoc(saliency, nullptr, nullptr, nullptr, &maximumLocation);

    return maximumLocation;
}

// Finds multiple high-saliency source pixels
// uses minimum-distance suppression to spread sources across different areas of the image
// greedy spacial suppression is used to prevent all the starting points for the algortihm from being contained in one small area
vector<Point> findSaliencySources(const Mat& saliency, double saliencyThreshold, int minDistance, int maxSources){
    struct Candidate{Point location; double score;};
    vector<Candidate> candidates;

    //collect pixels that exceed the given saliency threshold
    for (int y=0; y<saliency.rows; ++y){
        for (int x=0; x<saliency.cols; ++x){
            double score = saliency.at<double>(y, x);
            if(score >= saliencyThreshold){
                candidates.push_back({ Point(x, y), score });
            }
        }
    }
    // Sort candidates from highest to lowest saliency
    sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b){
        return a.score > b.score;
    });
    vector<Point> sources;

    // Select high saliency pixels while keeping some spacing between them
    for (const Candidate& candidate : candidates){
        bool tooClose = false;

        for (const Point& selected:sources){
            double dx = candidate.location.x - selected.x;
            double dy = candidate.location.y - selected.y;

            double distance = sqrt(dx * dx + dy * dy);
            if(distance < minDistance){
                tooClose = true;
                break;
            }
        }

        if (!tooClose){
            sources.push_back(candidate.location);
            if (static_cast<int>(sources.size()) >= maxSources){
                break;
            }
        }
    }
    return sources;
}

// creates heatmap from float saliencies
Mat createHeatmap(const Mat& saliency){
    Mat greyscale;
    Mat heatmap;

    saliency.convertTo(greyscale, CV_8U, 255.0);

    applyColorMap(greyscale, heatmap, COLORMAP_JET);

    return heatmap;
}

// Visualizes paths with low cost aka traversing through high saliency paths
Mat visualizePaths(const Mat& image, const vector<double>& distances, int source, double threshold){
    Mat output = image.clone();

    int width = image.cols;
    int height = image.rows;

    for (int node = 0; node < static_cast<int>(distances.size()); ++node){
        if (distances[node] <= threshold){
            Point p = nodeToPixel(node, width);

            //Marks pixels that are reached within cost threshold
            output.at<Vec3b>(p.y, p.x)=Vec3b(0,255,0);
        }
    }

    Point sourcePoint = nodeToPixel(source, width);

    circle(output, sourcePoint, 5, Scalar(0, 0, 255), FILLED);

    return output;
}

//Highlights pixels that exceed the threshold
Mat visualizeSalientRegions(const Mat& image, const Mat& saliency, double threshold){
    Mat output = image.clone();

    for (int y=0; y <image.rows; ++y){
        for (int x = 0; x < image.cols; ++x){
            double score = saliency.at<double>(y, x);

            if (score >= threshold){
                Vec3b& pixel = output.at<Vec3b>(y,x);

                // Blends the orginal pixel with red for visualization
                pixel[0] = static_cast<uchar>(pixel[0] * 0.5);
                pixel[1] = static_cast<uchar>(pixel[1] * 0.5);
                pixel[2] = static_cast<uchar>(pixel[2] * 0.5 + 127);
            }
        }
    }
    return output;
}

// using the combined saliency matrix the function identifies groups of pixels that have hight combined scores
// Using openCV's connectedComponentWithStats(), apply a threshold to the combined saliency map, convert result into a binary image
// identify connected components, remove very small regions, and visualize the remaining regions
Mat visualizeCombinedRegions(const Mat& image, const Mat& combinedSaliency, double threshold){
    Mat output = image.clone();

    // Convert combined saliency into a binary mask
    Mat binaryMask;

    threshold = max(0.0, min(.10, threshold));

    compare(combinedSaliency, threshold, binaryMask, CMP_GE);

    // smooth the binary mask slightly to reduce noise
    Mat smoothedMask;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(binaryMask, smoothedMask, MORPH_CLOSE, kernel);

    // Identify connected regions
    Mat labels;
    Mat stats;
    Mat centroids;

    int numberOfLabels = connectedComponentsWithStats(smoothedMask,labels, stats, centroids, 8, CV_32S);

    // Ignore very small regions
    int minimumArea = max(10, static_cast<int>(image.total() * 0.01));

    int regionCount = 0;
    for (int label = 1; label < numberOfLabels; ++label){
        int area = stats.at<int>(label, CC_STAT_AREA);

        if (area < minimumArea){
            continue;
        }

        int x = stats.at<int>(label, CC_STAT_LEFT);
        int y = stats.at<int>(label, CC_STAT_TOP);

        int width = stats.at<int>(label, CC_STAT_WIDTH);
        int height = stats.at<int>(label, CC_STAT_HEIGHT);

        regionCount++;

        // Color pixels belonging to this region
        for (int py = y; py < y + height; ++py){
            for (int px = x; px <x + width; ++px){
                if (labels.at<int>(py, px) == label){
                    Vec3b& pixel = output.at<Vec3b>(py, px);

                    // Blend region with red for visualization
                    pixel[0] = static_cast<uchar>(pixel[0] * 0.5);
                    pixel[1] = static_cast<uchar>(pixel[1] * 0.5);
                    pixel[2] = static_cast<uchar>(pixel[2] * 0.5 + 127);
                }
            }
        }
        // Creating a bouding box around the region
        rectangle(output, Rect(x, y, width, height), Scalar(0, 255, 0), 2);

        cout << "Region " << regionCount << " | Area: " << area << " pixels" << endl;
    }
    cout << "Total salient regions identified: " << regionCount << endl;

    // returns connected regions that satisfy conbined saliency threshold
    // connects adjacent pixels not necessarily complete objects if two objects touch they may become one region
    return output;
}

// Create a combined heatmap from combined saliency scores
Mat createCombinedHeatmap(const Mat& combinedSaliency){
    Mat grayscale;
    Mat heatmap;

    combinedSaliency.convertTo(grayscale, CV_8U, 255.0);
    applyColorMap(grayscale, heatmap, COLORMAP_JET);
    return heatmap;
}

// Visualize all selected Dijkstra source pixels
Mat visualizeSources(const Mat& image, const vector<Point>& sources){
    Mat output = image.clone();
    for (size_t i=0; i<sources.size(); ++i){
        //Draws a circle around each source
        circle(output, sources[i], 7, Scalar(0, 0, 255), FILLED);

        // Labels each source
        putText(output, to_string(i + 1), sources[i] + Point(8, -8), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }
    return output;
}


int main(int argc, char* argv[]){
    try {
        if (argc < 2){
            cerr << "Usage: " << argv[0] << " <image_path>" << endl;
            return 1;
        }

        // Loads input image
        Mat original = imread(argv[1], IMREAD_COLOR);
        if (original.empty()){
            cerr << "Error: Could not load image " << argv[1] << endl;
            return 1;
        }

        cout<< "Image loaded successfully" << endl;
        cout << "Original dimensions of image: " << original.cols << " x " << original.rows << endl;


        // resizes the image for computations
        const int maxDimension = 1000;
        double scale = min(1.0, static_cast<double>(maxDimension) / max(original.cols, original.rows));

        Mat image;
        resize( original, image, Size(), scale, scale, INTER_AREA);
        cout << "Processing dimensions: " << image.cols << " x " << image.rows << endl;


        // Calculating saliency
        Mat saliency = calculateSaliency(image);
        cout << "Saliency map calulcate" << endl;


        // Finding multiple high saliency source pixels
        cout << "Finding multiple saliency sources..." << endl;

        double sourceThreshold = 0.71; // How high the saliency score needs to be to be considered
        int minSourceDistance = 70; // Able to change how close or far the points can be centered
        int maxSources = 60; // How many different points are used

        vector<Point> sourcePoints = findSaliencySources(saliency, sourceThreshold, minSourceDistance, maxSources);
        if (sourcePoints.empty()){
            throw runtime_error("No saliency source pixels were found.");
        }

        // Converting source coordinates into graph vertex indices
        vector<int> sources;

        for (const Point& point : sourcePoints){
            int source = pixelToNode(point.x, point.y, image.cols);
            sources.push_back(source);
            cout << "Source " << sources.size() << ": (" << point.x << "," << point.y << ") | Saliency: " << saliency.at<double>(point.y, point.x) << endl;
        }
        cout << "Total sources selected: " << sources.size() << endl;


        // Building Graph
        cout << "Building Graph" << endl;
        vector<vector<Edge>> graph = buildGraph(image, saliency);
        cout << "Graph contains " << graph.size() << " verticies." << endl;


        // Runing D's
        cout << "Running multi source Dijkstra's algorithm" << endl;
        vector<double> distances = dijkstra(graph, sources);
        cout << "Completed multi source Dijkstra's" << endl;

        // Calculating combined saliency using Dijkstra distances
        cout << "Calculating combined saliency..." << endl;

        double tau = 75.0; // controls how quick dijkstra influence decreases
        cout << "Tau" << tau << endl;
       

        // Visualize seleced dource pixels
        Mat sourceImage = visualizeSources(image, sourcePoints);

        // Creates original saliency heatmap
        Mat heatmap = createHeatmap(saliency);

        // Creates combined saliency
        Mat combinedSaliency = calculateCombinedSaliency(saliency, distances, tau);
         cout << "Combined saliency calculated." << endl;

        //create combined saliency heatmap
        Mat combinedHeatmap = createCombinedHeatmap(combinedSaliency);

        // Identify regions using combined saliency
        cout << "Identifying salient regions..." << endl;

        // Threshold allows more or less pixels to qualify as part of a salient region
        Mat regionImage = visualizeCombinedRegions(image, combinedSaliency, 0.10);

        cout << "Region identification completed" << endl;

        // Save outputs
        imwrite("saliency_map.jpg", heatmap);
        imwrite("combined_saliency.jpg", combinedHeatmap);
        imwrite("combined_regions.jpg", regionImage);
        imwrite("selected_sources.jpg", sourceImage);

        cout << "\nResults saved:" << endl;
        cout << "saliency_map.jpg" << endl;
        cout << "combined_saliency.jpg" << endl;
        cout << "combined_regions.jpg" << endl;

        // Display outputs
        imshow("Original Image", image);
        imshow("Original Saliency", heatmap);
        imshow("Combined Saliency - Dijkstra", combinedHeatmap);
        imshow("Identified Salient Regions", regionImage);

        waitKey(0);

        return 0;

    }catch (const exception& error) {
        cerr << "Error: " << error.what() << endl;
        return 1;
    }
}
*/
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct QueueNode {
    int node;
    double distance;
    bool operator>(const QueueNode& other) const { return distance > other.distance; }
};

struct Candidate { Point location; double score; };

static int pixelToNode(int x, int y, int width) { return y * width + x; }
static double colorDistance(const Vec3b& a, const Vec3b& b);

// Local contrast used to propose source points; later stages add explicit black-frame evidence.
static Mat calculateSaliency(const Mat& image) {
    Mat f, blurred, difference, saliency;
    image.convertTo(f, CV_32FC3, 1.0 / 255.0);
    GaussianBlur(f, blurred, Size(0, 0), 7.0);
    absdiff(f, blurred, difference);
    vector<Mat> channels;
    split(difference, channels);
    Mat squared = channels[0].mul(channels[0]) + channels[1].mul(channels[1]) + channels[2].mul(channels[2]);
    sqrt(squared, saliency);
    normalize(saliency, saliency, 0.0, 1.0, NORM_MINMAX);
    return saliency;
}

// Estimate black-frame likelihood in [0,1]. White-background pixels score near zero.
static Mat blackEvidence(const Mat& image) {
    Mat f, hsv, darkness, saturation, evidence;
    image.convertTo(f, CV_32FC3, 1.0 / 255.0);
    cvtColor(image, hsv, COLOR_BGR2HSV);
    vector<Mat> ch;
    split(hsv, ch);
    // A smooth darkness ramp avoids a brittle hard threshold and tolerates highlights on black plastic.
    Mat value;
    ch[2].convertTo(value, CV_32F, 1.0 / 255.0);
    darkness = (0.68f - value) / 0.38f;
    cv::max(darkness, 0.0, darkness);
    cv::min(darkness, 1.0, darkness);
    ch[1].convertTo(saturation, CV_32F, 1.0 / 255.0);
    // Dark neutral pixels receive strongest evidence; saturated dark colors are downweighted slightly.
    evidence = darkness.mul(1.0f - 0.20f * saturation);
    GaussianBlur(evidence, evidence, Size(3, 3), 0.6);
    return evidence;
}

static Mat buildSeedScore(const Mat& saliency, const Mat& black) {
    Mat score = saliency.mul(black);
    normalize(score, score, 0.0, 1.0, NORM_MINMAX);
    return score;
}

// Keep only spatial local maxima, then greedy non-maximum suppression spreads seeds apart.
static vector<Point> findLocalMaximumSources(const Mat& score, const Mat& black,
                                             double minScore, double minBlack,
                                             int minDistance, int maxSources) {
    Mat dilated;
    dilate(score, dilated, getStructuringElement(MORPH_ELLIPSE, Size(9, 9)));
    vector<Candidate> candidates;
    for (int y = 0; y < score.rows; ++y) for (int x = 0; x < score.cols; ++x) {
        const double s = score.at<float>(y, x);
        if (s >= minScore && black.at<float>(y, x) >= minBlack && s >= dilated.at<float>(y, x) - 1e-6)
            candidates.push_back({Point(x, y), s});
    }
    sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    vector<Point> sources;
    for (const auto& c : candidates) {
        bool separated = true;
        for (Point p : sources) if ((p - c.location).dot(p - c.location) < minDistance * minDistance) { separated = false; break; }
        if (separated) sources.push_back(c.location);
        if (static_cast<int>(sources.size()) >= maxSources) break;
    }
    // Flat or very clean images may have no qualifying local peak; relax the score cutoff,
    // while still requiring dark-frame evidence and spatial suppression.
    if (sources.empty()) {
        for (const auto& c : candidates) {
            (void)c;
        }
        double best = 0.0;
        for (int y=0; y<score.rows; ++y) for (int x=0; x<score.cols; ++x)
            if (black.at<float>(y,x) >= minBlack) best = max(best, static_cast<double>(score.at<float>(y,x)));
        for (int y=0; y<score.rows; ++y) for (int x=0; x<score.cols; ++x) {
            double s=score.at<float>(y,x);
            if (s < best * 0.55 || black.at<float>(y,x) < minBlack || s < dilated.at<float>(y,x)-1e-6) continue;
            bool separated=true;
            for (Point p:sources) if ((p-Point(x,y)).dot(p-Point(x,y)) < minDistance*minDistance) { separated=false; break; }
            if (separated) sources.emplace_back(x,y);
            if (static_cast<int>(sources.size()) >= maxSources) break;
        }
    }
    return sources;
}

// Four-neighbour grid Dijkstra evaluated on demand to avoid storing millions of heap-allocated edges.
static vector<float> dijkstra(const Mat& image, const Mat& saliency, const Mat& black,
                              const vector<Point>& sources, double alpha, double beta) {
    const int w=image.cols, h=image.rows, n=w*h;
    vector<float> dist(n, numeric_limits<float>::infinity());
    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> pq;
    for (Point p:sources) { int id=pixelToNode(p.x,p.y,w); dist[id]=0; pq.push({id,0}); }
    const int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
    while (!pq.empty()) {
        auto cur=pq.top(); pq.pop();
        if (cur.distance > dist[cur.node]+1e-6) continue;
        int x=cur.node%w, y=cur.node/w;
        for (int k=0;k<4;++k) {
            int nx=x+dx[k], ny=y+dy[k]; if(nx<0||nx>=w||ny<0||ny>=h) continue;
            double color=colorDistance(image.at<Vec3b>(y,x), image.at<Vec3b>(ny,nx));
            // Cheap travel along dark pixels; expensive travel into the bright background.
            double darkness=(black.at<float>(y,x)+black.at<float>(ny,nx))*0.5;
            double cost=0.025 + alpha*(1.0-(saliency.at<float>(y,x)+saliency.at<float>(ny,nx))*0.5)
                        + beta*color + 1.8*(1.0-darkness);
            int next=pixelToNode(nx,ny,w); double nd=cur.distance+cost;
            if(nd<dist[next]) { dist[next]=static_cast<float>(nd); pq.push({next,nd}); }
        }
    }
    return dist;
}

static double colorDistance(const Vec3b& a, const Vec3b& b) {
    double db=double(a[0])-b[0], dg=double(a[1])-b[1], dr=double(a[2])-b[2];
    return sqrt(db*db+dg*dg+dr*dr)/(255.0*sqrt(3.0));
}

static Mat makeMask(const Mat& black, const vector<float>& distances, double pathTau,
                    double blackThreshold, double pathThreshold, int closeSize) {
    Mat mask(black.size(), CV_8U, Scalar(0));
    for(int y=0;y<black.rows;++y) for(int x=0;x<black.cols;++x) {
        double path=exp(-distances[pixelToNode(x,y,black.cols)]/pathTau);
        // Frame evidence is the primary selector. A modest path term bridges narrow gaps along the frame.
        if(black.at<float>(y,x)>=blackThreshold && path>=pathThreshold) mask.at<uchar>(y,x)=255;
    }
    int k=max(3,closeSize|1);
    morphologyEx(mask,mask,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(k,k)));
    return mask;
}

static Mat visualizeRegions(const Mat& image, const Mat& mask, int minArea, int& count) {
    Mat labels, stats, centroids;
    count=connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
    Mat out=image.clone(); int kept=0;
    for(int label=1;label<count;++label) {
        int area=stats.at<int>(label,CC_STAT_AREA); if(area<minArea) continue;
        ++kept;
        for(int y=0;y<labels.rows;++y) for(int x=0;x<labels.cols;++x) if(labels.at<int>(y,x)==label) {
            Vec3b& p=out.at<Vec3b>(y,x);
            p[0]=static_cast<uchar>(0.65*p[0]); p[1]=static_cast<uchar>(0.65*p[1]+24); p[2]=static_cast<uchar>(0.65*p[2]+89);
        }
        Rect box(stats.at<int>(label,CC_STAT_LEFT),stats.at<int>(label,CC_STAT_TOP),
                 stats.at<int>(label,CC_STAT_WIDTH),stats.at<int>(label,CC_STAT_HEIGHT));
        rectangle(out,box,Scalar(0,255,0),2);
        cout<<"Region "<<kept<<" | area "<<area<<" px | box "<<box<<endl;
    }
    count=kept; return out;
}

static Mat colorize(const Mat& src) {
    Mat u8,out; src.convertTo(u8,CV_8U,255.0); applyColorMap(u8,out,COLORMAP_JET); return out;
}

int main(int argc,char** argv) {
    try {
        if(argc<2) { cerr<<"Usage: "<<argv[0]<<" <image_path> [output_directory]\n"; return 1; }
        Mat original=imread(argv[1],IMREAD_COLOR);
        if(original.empty()) throw runtime_error(string("Could not load image: ")+argv[1]);
        const string outDir=argc>=3?argv[2]:".";
        const int maxDimension=900;
        double scale=min(1.0,double(maxDimension)/max(original.cols,original.rows));
        Mat image; resize(original,image,Size(),scale,scale,INTER_AREA);
        cout<<"Processing "<<image.cols<<" x "<<image.rows<<"\n";

        // Tune these values against the saved debug maps.
        const double sourceThreshold=0.12, minBlackForSource=0.45;
        const int minSourceDistance=24, maxSources=40;
        const double alpha=0.20, beta=0.70; // saliency and color-boundary contributions to edge cost
        const double pathTau=28.0;           // path-cost decay length; increase to spread farther
        const double blackThreshold=0.48, pathThreshold=0.10;
        const int closingKernel=9, minimumRegionArea=max(12,int(image.total()*0.00008));

        Mat saliency=calculateSaliency(image), black=blackEvidence(image);
        Mat seedScore=buildSeedScore(saliency,black);
        vector<Point> points=findLocalMaximumSources(seedScore,black,sourceThreshold,minBlackForSource,minSourceDistance,maxSources);
        if(points.empty()) throw runtime_error("No dark-frame source maxima found; lower sourceThreshold or minBlackForSource.");
        cout<<"Selected "<<points.size()<<" local-maximum Dijkstra sources\n";
        Mat sourceView=image.clone(); for(size_t i=0;i<points.size();++i) {
            circle(sourceView,points[i],4,Scalar(0,0,255),FILLED);
            putText(sourceView,to_string(i+1),points[i]+Point(5,-4),FONT_HERSHEY_SIMPLEX,0.35,Scalar(0,255,0),1);
        }
        vector<float> distances=dijkstra(image,saliency,black,points,alpha,beta);
        Mat path(black.size(),CV_32F);
        for(int y=0;y<path.rows;++y) for(int x=0;x<path.cols;++x)
            path.at<float>(y,x)=exp(-distances[pixelToNode(x,y,path.cols)]/pathTau);
        Mat mask=makeMask(black,distances,pathTau,blackThreshold,pathThreshold,closingKernel);
        int regionCount=0; Mat regions=visualizeRegions(image,mask,minimumRegionArea,regionCount);
        Mat blackView; black.convertTo(blackView,CV_8U,255); // grayscale black-likelihood evidence
        Mat binaryView; applyColorMap(blackView,binaryView,COLORMAP_BONE);
        string sep=(outDir=="."||outDir.empty())?"/":"/";
        imwrite(outDir+sep+"saliency_map.jpg",colorize(saliency));
        imwrite(outDir+sep+"black_frame_evidence.jpg",binaryView);
        imwrite(outDir+sep+"selected_sources.jpg",sourceView);
        imwrite(outDir+sep+"path_influence.jpg",colorize(path));
        imwrite(outDir+sep+"glasses_mask.png",mask);
        imwrite(outDir+sep+"combined_regions.jpg",regions);
        cout<<"Saved saliency_map.jpg, black_frame_evidence.jpg, selected_sources.jpg, path_influence.jpg, glasses_mask.png, combined_regions.jpg to "<<outDir<<"\n";
        cout<<"Retained connected regions: "<<regionCount<<"\n";
        return 0;
    } catch(const exception& e) { cerr<<"Error: "<<e.what()<<"\n"; return 1; }
}
