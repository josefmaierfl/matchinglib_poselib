//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2021 Josef Maier
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

#include <utils_common.h>
#include <regex>
#include <cmath>
#include <iostream>

namespace utilslib
{
    std::string str_replace(const std::string &in, const std::vector<char> &from, const std::string &to)
    {
        std::string to_repl;
        bool nfirst = false;
        for (auto &c : from)
        {
            if (nfirst)
            {
                to_repl.append("|");
            }
            if (c == '.')
            {
                to_repl.append("\\.");
            }
            else
            {
                to_repl.push_back(c);
            }
            nfirst = true;
        }
        return std::regex_replace(in, std::regex(to_repl), to);
    }

    Statistics::Statistics(const std::vector<double> &vals)
    {
        if (vals.empty()){
            return;
        }
        nr_elements = vals.size();
        std::vector<double> vals_tmp(vals);

        std::sort(vals_tmp.begin(), vals_tmp.end(), [](double const &first, double const &second)
                  { return first < second; });

        min = vals_tmp[0];
        max = vals_tmp.back();

        if (nr_elements % 2)
            median = vals_tmp[(nr_elements - 1) / 2];
        else
            median = (vals_tmp[nr_elements / 2] + vals_tmp[nr_elements / 2 - 1]) / 2.0;

        getMeanStandardDeviation(vals_tmp, mean, sd);
    }

    void Statistics::scale(const double &s)
    {
        mean *= s;
        median *= s;
        sd *= s;
        min *= s;
        max *= s;
    }

    size_t binomialCoeff(const size_t &n, const size_t &k)
    {
        //Adapted version from: https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
        std::vector<size_t> C(k + 1, 0);

        C[0] = 1; // nC0 is 1

        for (size_t i = 1; i <= n; i++)
        {
            // Compute next row of pascal triangle using
            // the previous row
            for (size_t j = std::min(i, k); j > 0; j--)
                C[j] = C[j] + C[j - 1];
        }
        return C[k];
    }

    void printCvMat(const cv::Mat &m, const std::string &name)
    {
        std::cout << name << ":" << std::endl;
        const int type = m.type();
        for (int i = 0; i < m.rows; i++)
        {
            for (int j = 0; j < m.cols; j++)
            {
                if (type == CV_64FC1)
                {
                    std::cout << m.at<double>(i, j) << "  ";
                }
                else if (type == CV_32FC1)
                {
                    std::cout << m.at<float>(i, j) << "  ";
                }
                else if (type == CV_8UC1)
                {
                    std::cout << m.at<unsigned char>(i, j) << "  ";
                }
                else
                {
                    throw std::runtime_error("Type not supported");
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}