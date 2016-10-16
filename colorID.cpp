#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <assert.h>

using namespace std;
using namespace cv;

/*
string colorID(Mat& inputImage) {

    vector<Mat> rgb;
    split(inputImage, rgb); //Splits Mat into r g b mats
    int histNum = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat r_hist, g_hist, b_hist; //Instances of r g b histograms

    //Populating r g b histograms
    calcHist( &rgb[0], 1, 0, Mat(), b_hist, 1, &histNum, &histRange, uniform, accumulate );
    calcHist( &rgb[1], 1, 0, Mat(), g_hist, 1, &histNum, &histRange, uniform, accumulate );
    calcHist( &rgb[2], 1, 0, Mat(), r_hist, 1, &histNum, &histRange, uniform, accumulate );

    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histNum );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0) );

    //Normalize the r g b mats
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for(int i=1; i<histNum; i++) {

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0 ), 2, 8, 0);

        cout << cvRound(b_hist.at<float>(i-1)) << "," << cvRound(b_hist.at<float>(i)) << " ";

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
                         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                         Scalar( 0, 0, 255 ), 2, 8, 0);

        cout << cvRound(b_hist.at<float>(i-1)) << "," << cvRound(b_hist.at<float>(i)) << " ";

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
                         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0 ), 2, 8, 0);

        cout << cvRound(b_hist.at<float>(i-1)) << "," << cvRound(b_hist.at<float>(i)) << " ";
        cout << endl;
    }

    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
    imshow("calcHist Demo", histImage);

    waitKey(0);
}
*/

string colorID(Mat img) {
    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] = {
        Scalar(0, 0, 255),
        Scalar(0, 255, 0),
        Scalar(255, 100, 100),
        Scalar(255, 0, 255),
        Scalar(0, 255, 255)
    };

    RNG rng(12345);

    for(;;) {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int i, sampleCount = rng.uniform(1, 1001);
        Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        Mat centers;

        // Generate random sample from multigaussian distribution
        for( k=0; k<clusterCount; k++ ) {
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount-1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
        }

        randShuffle(points, 1, &rng);

        kmeans(points, clusterCount, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                3, KMEANS_PP_CENTERS, centers);

        img = Scalar::all(0);

        for( i=0; i<sampleCount; i++ ) {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
        }

        imshow("clusters", img);

        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // ESC
            break;
    }

}


int main(int argc, char**argv) {
    assert(argc > 1);

    string input(argv[1]);
    Mat im = imread(input);
    if(!im.data)
        return -1;
    cout << colorID(im) << endl;

    return 0;
}
