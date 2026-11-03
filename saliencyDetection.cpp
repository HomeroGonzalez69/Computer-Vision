#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

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
    const double epsilon = 0.01;
    const double alpha = 3.0;
    const double beta = 120.0;

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
        const int maxDimension = 2600;
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


        //Creating visualization
        //cout<<"creating heatmap" <<endl;
        Mat heatmap = createHeatmap(saliency);
        //cout<<"done with heatmap" <<endl;
        //cout<<"making path image" <<endl;
        Mat pathImage = visualizePaths(image, distances, source, 200.0);
        //cout<<"done with path image" <<endl;
        //cout<<"creating region image" <<endl;
        Mat regionImage = visualizeSalientRegions(image, saliency, 0.50);
        //cout<<"done with region image" <<endl;


        //saves outputs
        imwrite("saliency_map.jpg", heatmap);
        imwrite("dijkstra_paths.jpg", pathImage);
        imwrite("salient_regions.jpg", regionImage);

        cout << "\nResults saved:" << endl;
        cout << "  saliency_map.jpg" << endl;
        cout << "  dijkstra_paths.jpg" << endl;
        cout << "  salient_regions.jpg" << endl;


        // Display the results.
        imshow("Original Image", image);
        imshow("Saliency Heatmap", heatmap);
        imshow("Salient Regions", regionImage);
        imshow("Dijkstra Paths", pathImage);

        waitKey(0);

        return 0;

    }catch (const exception& error) {
        cerr << "Error: " << error.what() << endl;
        return 1;
    }
}
