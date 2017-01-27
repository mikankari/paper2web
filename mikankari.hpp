#include <iostream>
#include <stack>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/**
 * わいのねいむすぺーす。OpenCVでできなかったことを関数にした
 */

namespace mikankari{

	/**
	 * メディアンフィルタ。４近傍に対して適用されます
	 * 
	 * @param src	入力画像
	 * @param dst	出力画像
	 */
	
	void medianBlur(Mat &src, Mat &dst){
		dst.create(src.size(), src.type());
		
		for(int y; y < src.rows; y++){
			for(int x; x < src.cols; x++){
				Point nears[] = {
					Point(-1, 0),
					Point(0, -1),
					Point(0, 0),
					Point(0, 1),
					Point(1, 0)
				};
				Mat filtered = Mat::zeros(1, 5, CV_32S);
				for(int i = 0; i < 5; i++){
					Point near = Point(x, y) + nears[i];
//					if(near.y >= 0 && near.y < src.rows && near.x >= 0 && near.x < src.cols){
						Vec3b pixel = src.at<Vec3b>(near);
						filtered.at<Vec3b>(0, i) = pixel;
//					}
				}
				for(int i = 0; i < 5; i++){
					for(int j = 5 - 1; j > i; j--){
						Vec3b &pixel1 = filtered.at<Vec3b>(0, j);
						Vec3b &pixel2 = filtered.at<Vec3b>(0, j - 1);
						if(pixel1[0] + pixel1[1] + pixel1[2] < pixel2[0] + pixel2[1] + pixel2[2]){
							Vec3b escape = pixel1;
							pixel1 = pixel2;
							pixel2 = escape;
						}
					}
				}
				dst.at<Vec3b>(y, x) = filtered.at<Vec3b>(0, filtered.cols / 2);
			}
		}
	}

	/**
	 * ラベリング。カラー画像が使えます
	 *
	 * @param image	入力画像
	 * @param label	ラベル画像
	 * @param stats	ラベルの情報
	 * @param color	注目する色（省略時は白）
	 * @return ラベル数
	 */
	
	int connectedComponents(Mat &image, Mat &label, Mat &stats, Vec3b color = Vec3b(255, 255, 255)){
		label.create(image.size(), CV_32S);
		stats.create(1024, 6, CV_32S);
		
		int current = 0;
		stack<Point> waiting;
		int ksize = 3;
		for(int y = 0; y < image.rows; y++){
			for(int x = 0; x < image.cols; x++){
				if(label.at<int>(y, x) == 0){
					if(image.at<Vec3b>(y, x) == color){
						current++;

						int left = x;
						int top = y;
						int right = 0;
						int bottom = 0;
						int area = 1;

						label.at<int>(y, x) = current;
						waiting.push(Point(x, y));

						do{
							Point point = waiting.top();
							if(waiting.size() > 0){
								waiting.pop();
							}

							if(left > point.x){
								left = point.x;
							}
							if(top > point.y){
								top = point.y;
							}
							if(right < point.x){
								right = point.x;
							}
							if(bottom < point.y){
								bottom = point.y;
							}
							area++;

							for(int ty = -ksize / 2; ty <= ksize / 2; ty++){
								for(int tx = -ksize / 2; tx <= ksize / 2; tx++){
									Point near = point + Point(tx, ty);
									if(near.y >= 0 && near.y < image.rows && near.x >= 0 && near.x < image.cols){
										if(label.at<int>(near) == 0){
											if(image.at<Vec3b>(near) == color){
												label.at<int>(near) = current;
												waiting.push(near);
											}
										}
									}
								}
							}
						} while(waiting.size() > 0);

						stats.at<int>(current, 0) = left;
						stats.at<int>(current, 1) = top;
						stats.at<int>(current, 2) = right - left + 1;
						stats.at<int>(current, 3) = bottom - top + 1;
						stats.at<int>(current, 4) = area;
						stats.at<int>(current, 5) = 0;
						
					}
				}
			}
		}
		
		return current;
	}
	
	/**
	 * テンプレートマッチング。ROIを指定できます
	 * 
	 * @param image		画像１
	 * @param temol		画像２。テンプレート
	 * @param result		相違度画像
	 * @param matchedtempl	画像１と位置を揃えた画像２
	 * @param image_roi	画像１のROI
	 * @param templ_roi	画像２のROI
	*/

	void matchTemplate(Mat &image, Mat &templ, Mat &result, Mat &matchedtempl, Mat &image_roi, Mat &templ_roi){
		result.create(image.size(), image.type());

		Scalar min = Scalar(255 * image.size().area(), 255 * image.size().area(), 255 * image.size().area());
		Point minloc = Point(image.cols, image.rows);
		
		for(double angle = -0.5; angle <= -0.5; angle += 0.5){
			for(int y = -50; y <= 50; y++){
				for(int x = -50; x <= 50; x++){
					Mat matrix = getRotationMatrix2D(Point2f(image.cols / 2, image.rows / 2), angle, 1.0);
					matrix.at<double>(0, 2) += x;
					matrix.at<double>(1, 2) += y;

					Mat templ_transform;
					warpAffine(templ, templ_transform, matrix, templ_transform.size());
					Mat templ_roi_transform;
					warpAffine(templ_roi, templ_roi_transform, matrix, templ_roi_transform.size());

					Scalar current = Scalar(0, 0, 0);
					for(int ty = 0; ty < templ.rows; ty++){
						for(int tx = 0; tx < templ.cols; tx++){
							if(image_roi.at<Vec3b>(ty, tx) != Vec3b(0, 0, 0) && templ_roi.at<Vec3b>(ty, tx) == Vec3b(0, 0, 0)){
								Vec3b image_pixel = image.at<Vec3b>(ty, tx);
								Vec3b templ_pixel = templ_transform.at<Vec3b>(ty, tx);
								Vec3b diff_pixel = image_pixel - templ_pixel;
								current += Scalar(diff_pixel[0], diff_pixel[1], diff_pixel[2]);
								if(current[0] + current[1] + current[2] > min[0] + min[1] + min[2]){
									tx = templ.cols;
									ty = templ.rows;
								}
							}
						}
					}
//					result.at<Scalar>(y, x) = current;
					
					if(min[0] + min[1] + min[2] > current[0] + current[1] + current[2]){
						min = current;
						minloc = Point(x, y);
						matchedtempl = templ_transform;
						result = image - matchedtempl;
						result = result.mul(templ_roi_transform);
//						cout << angle << "deg " << minloc << ": " << min << endl;
					}
				}
				
				cout << "\r" << (y + 50) << "% " << flush;
			}
		}
		
		cout << endl;
	}
	
	/*
	 * ラベリング結果を渡して、近いところをつなげた結果を返します。
	 * 面積が小さいものはノイズとみなしてつなげません。
	 * 
	 * @param label	ラベル画像
	 * @param stats	ラベルの情報（行は各ラベル、列はx, y, width, height, areaの順）
	 * @param lines	つなげた新しいラベルの情報
	 * @param mode	つなげる方向。HORIZONTAL・VERTICAL・SLANTINGのいずれか
	 */
	
	const int HORIZONTAL = 0;
	const int VERTICAL = 1;
	const int SLANTING = 2;
	const int UNLIMITED = -1;
	
	void connectedLine(Mat &label, Mat &stats, Mat &lines, int mode, int min_area = 8, int max_area = 10000){
		Mat line_label = Mat::zeros(stats.rows, 3, CV_32S);
//		lines.create(stats.size(), stats.type());
		lines = Mat::zeros(stats.size(), stats.type());
		if(max_area < 0){
			max_area = label.rows * label.cols;
		}
		
		int current = 0;
		stack<int> waiting;

		for(int i = 0; i < stats.rows; i++){
			int left = stats.at<int>(i, 0);
			int top = stats.at<int>(i, 1);
			int width = stats.at<int>(i, 2);
			int height = stats.at<int>(i, 3);
			int area = stats.at<int>(i, 4);
			if(area >= min_area && area <= max_area/* && height < 20*/){

				if(line_label.at<int>(i, 0) == 0){
					current++;

//					cout << current << endl;

					int line_left = left;
					int line_top = top;
					int line_right = left + width;
					int line_bottom = top + height;
					int line_area = 0;

					int next = i;
					waiting.push(i);

					do{

						next = waiting.top();
						if(waiting.size() > 0){
							waiting.pop();
						}

//						cout << "   " << next << " waiting: " << waiting.size() << endl;

						left = stats.at<int>(next, 0);
						top = stats.at<int>(next, 1);
						width = stats.at<int>(next, 2);
						height = stats.at<int>(next, 3);
						area = stats.at<int>(next, 4);

//						if(area < 8 && area > 1000){
//							break;
//						}

						if(line_left > left){
							line_left = left;
						}
						if(line_top > top){
							line_top = top;
						}
						if(line_right < left + width){
							line_right = left + width;
						}
						if(line_bottom < top + height){
							line_bottom = top + height;
						}
						line_area += area;

						line_label.at<int>(next, 0) = current;

						if(mode == HORIZONTAL){
							/* 右端 */
							next = -1;
							for(int ty = -5; ty <= height + 5; ty++){
								for(int tx = width; tx <= width + 20; tx++){
									Point near = Point(left, top) + Point(tx, ty);
									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
										int index = label.at<int>(near);
										int label_area = stats.at<int>(index, 4);
										int label_height = stats.at<int>(index, 3);
										if(index != 0 && label_area >= 8 && label_area <= 10000/* && label_height < 20*/){
											if(line_label.at<int>(index, 0) == 0){
												next = index;
												waiting.push(index);

											}

											/* break loops */
//											tx = width + 41;
//											ty = height + 6;
										}
									}
								}
							}
//							cout << "      right: " << next << endl;

							/* 左端 */
							next = -1;
							for(int ty = -5; ty <= height + 5; ty++){
								for(int tx = -20; tx <= -1; tx++){
									Point near = Point(left, top) + Point(tx, ty);
									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
										int index = label.at<int>(near);
										int label_area = stats.at<int>(index, 4);
										int label_height = stats.at<int>(index, 3);
										if(index != 0 && label_area >= 8 && label_area <= 10000/* && label_height < 20*/){
											if(line_label.at<int>(index, 0) == 0){
												next = index;
												waiting.push(index);
											}

											/* break loops */
//											tx = 0;
//											ty = height + 6;
										}
									}
								}
							}
//							cout << "      left: " << next << endl;
							
						}else if(mode == VERTICAL){
//							/* 上端 */
//							next = -1;
//							for(int ty = -40; ty <= -1; ty++){
//								for(int tx = -5; tx <= width + 5; tx++){
//									Point near = Point(left, top) + Point(tx, ty);
//									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
//										int index = label.at<int>(near);
//										int label_area = stats.at<int>(index, 4);
//										int label_height = stats.at<int>(index, 3);
//										if(index != 0 && label_area >= min_area && label_area <= max_area/* && label_height < 20*/){
//											if(line_label.at<int>(index, 0) == 0){
//												next = index;
//												waiting.push(index);
//											}
//
//											/* break loops */
////											tx = width + 6;
////											ty = 0;
//										}
//									}
//								}
//							}
////						cout << "      top: " << next << endl;
//							
//							/* 下端 */
//							next = -1;
//							for(int ty = height + 1; ty <= height + 40; ty++){
//								for(int tx = -5; tx <= width + 5; tx++){
//									Point near = Point(left, top) + Point(tx, ty);
//									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
//										int index = label.at<int>(near);
//										int label_area = stats.at<int>(index, 4);
//										int label_height = stats.at<int>(index, 3);
//										if(index != 0 && label_area >= min_area && label_area <= max_area/* && label_height < 20*/){
//											if(line_label.at<int>(index, 0) == 0){
//												next = index;
//												waiting.push(index);
//											}
//
//											/* break loops */
////											tx = width + 6;
////											ty = height + 41;
//										}
//									}
//								}
//							}
////						cout << "      bottom: " << next << endl;
							
							/* 上下の周辺 */
							next = -1;
							for(int ty = -40; ty <= height + 40; ty++){
								for(int tx = -5; tx <= width + 5; tx++){
									Point near = Point(left, top) + Point(tx, ty);
									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
										int index = label.at<int>(near);
										int label_area = stats.at<int>(index, 4);
										int label_height = stats.at<int>(index, 3);
										if(index != 0 && label_area >= min_area && label_area <= max_area/* && label_height < 20*/){
											if(line_label.at<int>(index, 0) == 0){
												next = index;
												waiting.push(index);
											}

											/* break loops */
//											tx = width + 6;
//											ty = height + 41;
										}
									}
								}
							}
//							cout << "      bottom: " << next << endl;
						}else if(mode == SLANTING){
							/* 周辺 */
//							next = -1;
							for(int ty = -5; ty <= height + 5; ty++){
								for(int tx = -20; tx <= width + 20; tx++){
									Point near = Point(left, top) + Point(tx, ty);
									if(near.y >= 0 && near.y < label.rows && near.x >= 0 && near.x < label.cols){
										int index = label.at<int>(near);
										int label_area = stats.at<int>(index, 4);
										int label_height = stats.at<int>(index, 3);
										if(index != 0 && index != next && label_area >= min_area && label_area <= max_area /*&& label_height < 20*/){
											if(line_label.at<int>(index, 0) == 0){
//												next = index;
												waiting.push(index);
											}

											/* break loops */
//											tx = width + 6;
//											ty = height + 41;
										}
									}
								}
							}
//							cout << "      near: " << next << endl;
							
						}

					}while(waiting.size() > 0);

					lines.at<int>(current, 0) = line_left;
					lines.at<int>(current, 1) = line_top;
					lines.at<int>(current, 2) = line_right - line_left;
					lines.at<int>(current, 3) = line_bottom - line_top;
					lines.at<int>(current, 4) = line_area;

				}
			}
		}
		
	}
	
	void rectangle(Mat &image, Point lefttop, Point rightbottom, int color){
		for(int y = lefttop.y; y <= rightbottom.y; y++){
			for(int x = lefttop.x; x <= rightbottom.x; x++){
				if(y >= 0 && y < image.rows && x >= 0 && x < image.cols){
//					if(color < 256){
						image.at<int>(y, x) = color;
//						image.at<Vec3b>(y, x) = Vec3b(color, 0, 0);
//					}else{
//						image.at<Vec3b>(y, x) = Vec3b(color / 256, color % 256, 0);
//					}
				}
			}
		}
	}

}
