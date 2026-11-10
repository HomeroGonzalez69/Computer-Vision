#include <iostream>
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


        // Finding the most salient pixel
        Point sourcePoint = findMostSalientPixel(saliency);
        int source = pixelToNode(sourcePoint.x, sourcePoint.y, image.cols);
        cout << "Most salient pixel: (" << sourcePoint.x << ", " << sourcePoint.y << ")" << endl;
        cout<< "Maximum saliency score: " << saliency.at<double>(sourcePoint.y, sourcePoint.x) << endl;


        // Building Graph
        cout << "Building Graph" << endl;
        vector<vector<Edge>> graph = buildGraph(image, saliency);
        cout << "Graph contains " << graph.size() << " verticies." << endl;


        // Runing D's
        cout << "Running Dijkstra's algorithm" << endl;
        vector<double> distances = dijkstra(graph, source);
        cout << "Completed Dijkstra's" << endl;

        // Calculating combined saliency using Dijkstra distances
        cout << "Calculating combined saliency..." << endl;

        double tau = 75.0; // controls how quick dijkstra influence decreases
        cout << "Tau" << tau << endl;

        Mat combinedSaliency = calculateCombinedSaliency(saliency, distances, tau);

        cout << "Combined saliency calculated." << endl;

        // Creates original saliency heatmap
        Mat heatmap = createHeatmap(saliency);

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
