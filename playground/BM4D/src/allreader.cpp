/*
 * 2016, Vladislav Tananaev
 * v.d.tananaev [at] gmail [dot] com
 */
#include <bm4d-gpu/allreader.h>

#ifndef idx3
#define idx3(x, y, z, x_size, y_size) ((x) + ((y) + (y_size) * (z)) * (x_size))
#endif

void eat_comment(std::ifstream& f) {
    char linebuf[1024];
    char ppp;
    while (ppp = f.peek(), ppp == '\n' || ppp == '\r') f.get();
    if (ppp == '#') f.getline(linebuf, 1023);
}

void AllReader::readVideo(const std::string& filename, std::vector<unsigned char>& volume, int& width, int& height, int& depth) {
    cv::VideoCapture capture(filename);

    width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    depth = capture.get(CV_CAP_PROP_FRAME_COUNT);

    volume.resize(width * height * depth);

    cv::Mat frame;

    if (!capture.isOpened()) throw "Error when reading file";

    for (int slice = 0;; slice++) {
        if (!capture.read(frame)) break;

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                volume[idx3(x, y, slice, width, height)] = frame.at<unsigned char>(y, x);

        if (this->display) {
            cv::imshow("window", frame);
            char key = cvWaitKey(10);
            if (key == 27) break;    // ESC
        }
    }
}

void AllReader::read(const std::string& filename, std::vector<unsigned char>& volume, int& width, int& height, int& depth) {
    // Determine extension
    std::string ext = filename;
    ext.erase(0, ext.find_last_of('.'));
    if (ext == ".avi" || ext == ".AVI") {
        readVideo(filename, volume, width, height, depth);
    } else {
        std::cerr << "Unknown file extension: *" << ext << ". Available are .avi" << std::endl;
        exit(EXIT_FAILURE);
    }
}
