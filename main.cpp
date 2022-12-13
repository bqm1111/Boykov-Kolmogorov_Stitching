#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <vector>

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

typedef Graph<int, int, int> GraphType;

int main(int argc, char **argv)
{
    Mat A = imread(argv[1]);
    assert(A.data);

    Mat B;

    B = imread(argv[2]);
    assert(B.data);

    assert(A.rows == B.rows);

    Mat graphcut;
    Mat graphcut_and_cutline;

    int overlap_width = atoi(argv[3]);
    int xoffset = A.cols - overlap_width;

    Mat no_graphcut(A.rows, A.cols + B.cols - overlap_width, A.type());

    A.copyTo(no_graphcut(Rect(0, 0, A.cols, A.rows)));
    B.copyTo(no_graphcut(Rect(xoffset, 0, B.cols, B.rows)));

    int est_nodes = A.rows * overlap_width;
    int est_edges = est_nodes * 4;

    GraphType g(est_nodes, est_edges);

    for (int i = 0; i < est_nodes; i++)
    {
        g.add_node();
    }

    // Set the source/sink weights
    for (int y = 0; y < A.rows; y++)
    {
        g.add_tweights(y * overlap_width + 0, INT_MAX, 0);
        g.add_tweights(y * overlap_width + overlap_width - 1, 0, INT_MAX);
    }

    // Set edge weights
    for (int y = 0; y < A.rows; y++)
    {
        for (int x = 0; x < overlap_width; x++)
        {
            int idx = y * overlap_width + x;

            Vec3b a0 = A.at<Vec3b>(y, xoffset + x);
            Vec3b b0 = B.at<Vec3b>(y, x);
            double cap0 = norm(a0, b0);

            // Add right edge
            if (x + 1 < overlap_width)
            {
                Vec3b a1 = A.at<Vec3b>(y, xoffset + x + 1);
                Vec3b b1 = B.at<Vec3b>(y, x + 1);

                double cap1 = norm(a1, b1);

                g.add_edge(idx, idx + 1, (int)(cap0 + cap1), (int)(cap0 + cap1));
            }

            // Add bottom edge
            if (y + 1 < A.rows)
            {
                Vec3b a2 = A.at<Vec3b>(y + 1, xoffset + x);
                Vec3b b2 = B.at<Vec3b>(y + 1, x);

                double cap2 = norm(a2, b2);

                g.add_edge(idx, idx + overlap_width, (int)(cap0 + cap2), (int)(cap0 + cap2));
            }
        }
    }

    int flow = g.maxflow();
    // cout << "max flow: " << flow << endl;

    graphcut = no_graphcut.clone();
    graphcut_and_cutline = no_graphcut.clone();

    int idx = 0;
    for (int y = 0; y < A.rows; y++)
    {
        for (int x = 0; x < overlap_width; x++)
        {
            if (g.what_segment(idx) == GraphType::SOURCE)
            {
                graphcut.at<Vec3b>(y, xoffset + x) = A.at<Vec3b>(y, xoffset + x);
            }
            else
            {
                graphcut.at<Vec3b>(y, xoffset + x) = B.at<Vec3b>(y, x);
            }

            graphcut_and_cutline.at<Vec3b>(y, xoffset + x) = graphcut.at<Vec3b>(y, xoffset + x);

            // Draw the cut
            if (x + 1 < overlap_width)
            {
                if (g.what_segment(idx) != g.what_segment(idx + 1))
                {
                    graphcut_and_cutline.at<Vec3b>(y, xoffset + x) = Vec3b(0, 0255, 0);
                    graphcut_and_cutline.at<Vec3b>(y, xoffset + x + 1) = Vec3b(0, 255, 0);
                    graphcut_and_cutline.at<Vec3b>(y, xoffset + x - 1) = Vec3b(0, 255, 0);
                }
            }

            // Draw the cut
            if (y > 0 && y + 1 < A.rows)
            {
                if (g.what_segment(idx) != g.what_segment(idx + overlap_width))
                {
                    graphcut_and_cutline.at<Vec3b>(y - 1, xoffset + x) = Vec3b(0, 255, 0);
                    graphcut_and_cutline.at<Vec3b>(y, xoffset + x) = Vec3b(0, 255, 0);
                    graphcut_and_cutline.at<Vec3b>(y + 1, xoffset + x) = Vec3b(0, 255, 0);
                }
            }

            idx++;
        }
    }

    cv::imwrite("graphcut.jpg", graphcut);
    cv::imwrite("graphcut_and_cut_line.jpg", graphcut_and_cutline);

    cv::imshow("graphcut", graphcut);
    cv::imshow("graphcut and cut line", graphcut_and_cutline);
    cv::waitKey();
    return 0;
}
