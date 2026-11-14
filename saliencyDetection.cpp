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
// primary container for OpenCV
using namespace cv;

// Struct for running the D's Algo
struct QueueNode {
    int node;
    double distance;
    // forces a priority queue to act like min heap utilizing the distance between the nodes
    bool operator>(const QueueNode& other) const { return distance > other.distance; }
};

// Assigns candidate nodes scores
struct Candidate {
    Point location; 
    double score; 
};

/// @brief Converts image coordinates to graph vertex index
/// @param x 
/// @param y 
/// @param width 
/// @return 
static int pixelToNode(int x, int y, int width) { 
    return y * width + x; 
}

/// @brief creates two vectors of 3 bytes to compute the distances between colors (RGB)
/// @param a 
/// @param b 
/// @return 
static double colorDistance(const Vec3b& a, const Vec3b& b);

/// @brief  Local contrast is used to propose source points, later stages add explicit black-frames.
/// @param image 
/// @return Normalized saliency calculations 
static Mat calculateSaliency(const Mat& image) {
    // Creates the following Mat variables 
    Mat f, blurred, difference, saliency;

    // Converts image from 8 bit ints to a 3 Channel Matrix of 32bit floats stored within f protects from overflow errors that can come from an int
    image.convertTo(f, CV_32FC3, 1.0 / 255.0);
    // Applies a Gaussian blur to reduce noise within the image and allow an easier differenciation between objects, that new image is stored in blurred
    GaussianBlur(f, blurred, Size(0, 0), 7.0);
    // Takes the absdiff from blurred and f and stores it in difference. Diff now stores how strongly a specific color stands out from its surrounding pixels
    // Pixels that blend together ie sky pixels have a low difference score while edges with strong color boundries have high scores.
    absdiff(f, blurred, difference);

    vector<Mat> channels;
    // Splits the difference inherited 3 channel matricies within into RBG
    split(difference, channels);
    Mat squared = channels[0].mul(channels[0]) + channels[1].mul(channels[1]) + channels[2].mul(channels[2]);

    // stores the the overall color difference magnitude for each pixel across the three channels and stores it into a flat grayscale intensity map
    sqrt(squared, saliency);
    normalize(saliency, saliency, 0.0, 1.0, NORM_MINMAX);
    return saliency;
}

/// @brief Estimates black-frame likelihood in [0,1], background pixels score near zero.
/// @param image 
/// @return 
static Mat blackEvidence(const Mat& image) {
    // Creates empty Mat variables
    Mat f, hsv, darkness, saturation, evidence;
    image.convertTo(f, CV_32FC3, 1.0 / 255.0);

    // Converts image from RBG to Hue Saturation Value and splits the channels
    cvtColor(image, hsv, COLOR_BGR2HSV);
    vector<Mat> ch;
    split(hsv, ch);
    // A smooth darkness ramp avoids a brittle hard threshold and tolerates highlights on black plastic.
    Mat value;

    // Manipulates the second channel ie 0 (completely dark) -> 255 (max brightness)
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

/// @brief Builds numerical confidence assigned to candidate pixels
/// @param saliency 
/// @param black 
/// @return Score used to rank the most prominent pixels in the image
static Mat buildSeedScore(const Mat& saliency, const Mat& black) {
    Mat score = saliency.mul(black);
    normalize(score, score, 0.0, 1.0, NORM_MINMAX);
    return score;
}

/// @brief Keeps only spatial local maxima, then greedy non-maximum suppression spreads seeds apart to help create a clean bounding box per object.
/// @param score 
/// @param black 
/// @param minScore 
/// @param minBlack 
/// @param minDistance 
/// @param maxSources 
/// @return Sources vector of local Maxima in the image that passed the various filters
static vector<Point> findLocalMaximumSources(const Mat& score, const Mat& black, double minScore, double minBlack, int minDistance, int maxSources) {
    Mat dilated;
    // Creates a kernel of size 9x9 to start morphologicalling dilating the image and filtering out the local maxima
    // A ellipse is chosen to reduce sharp corner artifacts. Dilation expands peak pixel intensities so they can be more easily isolated
    dilate(score, dilated, getStructuringElement(MORPH_ELLIPSE, Size(9, 9)));

    // High confidence pixels are stored in the candidate vector from the source matrix in score
    vector<Candidate> candidates;
    for (int y = 0; y < score.rows; ++y) for (int x = 0; x < score.cols; ++x) {
        const double s = score.at<float>(y, x);

        // A pixels score must exceed a minimum threshold so background noise can be filtered out, the darkness value must also exceed the threshold
        // Uses an extreamly low tolerance so only the exact local peak is chosen as a candidate.
        if (s >= minScore && black.at<float>(y, x) >= minBlack && s >= dilated.at<float>(y, x) - 1e-6)
            candidates.push_back({Point(x, y), s});
    }

    // Candidates are sorted in order of their score
    sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    vector<Point> sources;

    // Ensures only the stronges candidates within certain distances are chosen to be points
    for (const auto& c : candidates) {
        bool separated = true;
        for (Point p : sources) if ((p - c.location).dot(p - c.location) < minDistance * minDistance) { separated = false; break; }
        if (separated) sources.push_back(c.location);
        if (static_cast<int>(sources.size()) >= maxSources) break;
    }

    // Flat or very clean images may have no qualifying local peaks so the score cutoff must be relaxed, while still requiring dark-frame evidence and spatial suppression. 
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

/// @brief Multi Source Dijkstra Algorithm to compute the pixels with the lowest travel cost from the starting seed points
/// @param image 
/// @param saliency 
/// @param black 
/// @param sources 
/// @param alpha 
/// @param beta 
/// @return Distance map between seed points
static vector<float> dijkstra(const Mat& image, const Mat& saliency, const Mat& black, const vector<Point>& sources, double alpha, double beta) {
    // Initializes various variables used to traverse the image and store the queue values
    const int w=image.cols, h=image.rows, n=w*h;
    // Fills the graph with infinity so the dijkstras can make sure to visit all the required points
    vector<float> dist(n, numeric_limits<float>::infinity());
    priority_queue<QueueNode, vector<QueueNode>, greater<QueueNode>> pq;

    for (Point p:sources) { int id=pixelToNode(p.x,p.y,w); dist[id]=0; pq.push({id,0}); }
    const int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
    // Pops the pixel with the lowest accumulated cost and checks its 4 immediate neighbors
    while (!pq.empty()) {
        auto cur=pq.top(); pq.pop();
        // Skips duplicated entries in the priority queue
        if (cur.distance > dist[cur.node]+1e-6) 
            continue;
        int x=cur.node%w, y=cur.node/w;

        // Utilizes custom parameters to determine how expensive it is to traverse from the seed pixel to the neighbor pixels
        // The more salient a pixel is the cheaper the traversal is, and crossing sharp color boundries is penalized.
        // Dark areas are cheaper than brighter areas.
        for (int k=0;k<4;++k) {
            int nx=x+dx[k], ny=y+dy[k]; if(nx<0||nx>=w||ny<0||ny>=h) continue;
            double color=colorDistance(image.at<Vec3b>(y,x), image.at<Vec3b>(ny,nx));

            // Cheap travel along dark pixels; expensive travel into the bright background.
            double darkness=(black.at<float>(y,x)+black.at<float>(ny,nx))*0.5;
            double cost=0.025 + alpha*(1.0-(saliency.at<float>(y,x)+saliency.at<float>(ny,nx))*0.5) + beta*color + 1.8*(1.0-darkness);
            int next=pixelToNode(nx,ny,w); double nd=cur.distance+cost;
            if(nd<dist[next]) { 
                dist[next]=static_cast<float>(nd); 
                pq.push({next,nd}); 
            }
        }
    }
    return dist;
}

/// @brief Calculates Eucladian distance between colors
/// @param a 
/// @param b 
/// @return Calculated Distance
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
    Mat u8,out; 
    src.convertTo(u8,CV_8U,255.0); 
    applyColorMap(u8,out,COLORMAP_JET); 
    return out;
}

int main(int argc,char** argv) {
    try {
        // Takes argument from the terminal
        if(argc<2) { cerr<<"Usage: "<<argv[0]<<" <image_path> [output_directory]\n"; return 1; }
        Mat original=imread(argv[1],IMREAD_COLOR);
        if(original.empty()) throw runtime_error(string("Could not load image: ")+argv[1]);
        const string outDir=argc>=3?argv[2]:".";
        const int maxDimension=900;
        double scale=min(1.0,double(maxDimension)/max(original.cols,original.rows));
        Mat image; resize(original,image,Size(),scale,scale,INTER_AREA);
        cout<<"Processing "<<image.cols<<" x "<<image.rows<<"\n";

        // Parameters that can be adjusted to optimize detection
        const double sourceThreshold=0.12, minBlackForSource=0.45;
        const int minSourceDistance=14, maxSources=35;
        const double alpha=0.20, beta=0.70; // saliency and color-boundary contributions to edge cost
        const double pathTau=28.0;           // path-cost decay length; increase to spread farther
        const double blackThreshold=0.48, pathThreshold=0.10;
        const int closingKernel=9, minimumRegionArea=max(12,int(image.total()*0.00008));

        // 
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
        
        // 
        Mat mask=makeMask(black,distances,pathTau,blackThreshold,pathThreshold,closingKernel);
        int regionCount=0; Mat regions=visualizeRegions(image,mask,minimumRegionArea,regionCount);
        Mat blackView; black.convertTo(blackView,CV_8U,255); // grayscale black-likelihood evidence
        Mat binaryView; applyColorMap(blackView,binaryView,COLORMAP_BONE);
        string sep=(outDir=="."||outDir.empty())?"/":"/";
        
        // Writes the files to .jpg for viewing and adds red to pixels for salient related images
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
