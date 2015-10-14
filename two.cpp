#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

//ofstream fout("data",ios_base::out | ios_base::app);
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
template<typename T> void printMat(const cv::Mat& img);

int main()
{
	Mat img(8,8,CV_8UC1,Scalar(0));
	img.ptr<uchar>(0)[1] = 1;
	img.ptr<uchar>(0)[2] = 1;
	img.ptr<uchar>(1)[2] = 1;
	img.ptr<uchar>(1)[3] = 1;
	img.ptr<uchar>(2)[1] = 1;
	img.ptr<uchar>(2)[2] = 1;
	img.ptr<uchar>(2)[3] = 1;
	img.ptr<uchar>(2)[5] = 1;
	img.ptr<uchar>(2)[6] = 1;
	img.ptr<uchar>(3)[0] = 1;
	img.ptr<uchar>(3)[1] = 1;
	img.ptr<uchar>(3)[4] = 1;
	img.ptr<uchar>(3)[5] = 1;
	img.ptr<uchar>(3)[6] = 1;
	img.ptr<uchar>(4)[1] = 1;
	img.ptr<uchar>(4)[2] = 1;
	img.ptr<uchar>(5)[2] = 1;
	img.ptr<uchar>(5)[4] = 1;
	img.ptr<uchar>(5)[5] = 1;
	img.ptr<uchar>(5)[6] = 1;
	img.ptr<uchar>(5)[7] = 1;
	img.ptr<uchar>(6)[2] = 1;
	img.ptr<uchar>(6)[3] = 1;
	img.ptr<uchar>(6)[5] = 1;
	img.ptr<uchar>(6)[7] = 1;
	img.ptr<uchar>(7)[0] = 1;

	printMat<uchar>(img);

	Mat label;
	icvprCcaByTwoPass(img,label);


	return 1;
}


void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	// connected component analysis (4-component)
	// use two-pass algorithm
	// 1. first pass: label each foreground pixel with a label
	// 2. second pass: visit each labeled pixel and merge neighbor labels
	// 
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return ;
	}

	// 1. first pass

	_lableImg.release() ;
	_binImg.convertTo(_lableImg, CV_32SC1) ;

	int label = 1 ;  // start by 2
	std::vector<int> labelSet ;
	labelSet.push_back(0) ;   // background: 0
	labelSet.push_back(1) ;   // foreground: 1

	int rows = _binImg.rows - 1 ;
	int cols = _binImg.cols - 1 ;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _lableImg.ptr<int>(i-1) ;
		int* data_curRow = _lableImg.ptr<int>(i) ;
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels ;
				neighborLabels.reserve(2) ;
				int leftPixel = data_curRow[j-1] ;
				int upPixel = data_preRow[j] ;
				if ( leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel) ;
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel) ;
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label) ;  // assign to a new label
					data_curRow[j] = label ;
					labelSet[label] = label ;
					//printMat(_lableImg);
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end()) ;
					int smallestLabel = neighborLabels[0] ;  
					data_curRow[j] = smallestLabel ;

					// save equivalence
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k] ;
						int& oldSmallestLabel = labelSet[tempLabel] ;
						if (oldSmallestLabel > smallestLabel)
						{							
							labelSet[oldSmallestLabel] = smallestLabel ;
							oldSmallestLabel = smallestLabel ;
						}						
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel ;
						}
					}
				}				
			}
		}

		printMat<int>(_lableImg);
	}

	// update equivalent labels
	// assigned with the smallest label in each equivalent label set
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int number = 0;
		int curLabel = labelSet[i] ;
		int preLabel = labelSet[curLabel] ;
		while (preLabel != curLabel)
		{
			number++;
			curLabel = preLabel ;
			preLabel = labelSet[preLabel] ;
		}
		labelSet[i] = curLabel ;
		cout << number;
	}


	// 2. second pass
	for (int i = 0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i) ;
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j] ;
			pixelLabel = labelSet[pixelLabel] ;	
		}
	}
}

 template<typename T> void printMat(const cv::Mat& img)
{
	int width = img.cols;
	int height = img.rows;
	cout << endl;
	for (int i = 0; i < height; i++)
	{
		const T *data = img.ptr<T>(i);
		for (int j = 0; j < width; j++)
			cout << (int)data[j] << " ";
		cout << endl;
	}
}
