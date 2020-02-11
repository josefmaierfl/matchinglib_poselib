//
// Created by maierj on 10.02.20.
//
#include "load_py_ngransac.hpp"

void ComputeServer::transferModel(ComputeInstance&, bp::list model, bp::list inlier_mask, unsigned int nr_inliers)
{
    // simulate sending an order, receiving an acknowledgement and calling back to the strategy instance

    std::cout << "sending order to market\n";
    std::vector<double> model_c = py_list_2_vec<double>(model);
    std::vector<bool> mask = py_list_2_vec<bool>(inlier_mask);
    Py_output out {nr_inliers, vecToMat(model_c, 3, 3), vecToMat(model_c, 1, (int)mask.size())};
//    Order order { symbol, side, size, price, ++_next_order_id };
}

