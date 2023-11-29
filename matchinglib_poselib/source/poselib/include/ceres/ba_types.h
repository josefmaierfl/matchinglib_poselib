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

#pragma once

namespace poselib
{
    enum class LossFunctionToUse
    {
        SQUARED,
        SOFT_L1,
        CAUCHY,
        HUBER
    };

    enum class MethodToUse
    {
        DEFAULT,                                                   // Use whatever is the default in each BA method
        LM_SparseNormalChol,                                       // Levenberg-Marquardt with sparse normal Choleski solver (exact linear solver)
        LM_SparseNormalChol_NonMonotonic,                          // Levenberg-Marquardt with sparse normal Choleski solver (exact linear solver) allowing a few non-montonic steps during iterations (allows to jump over boulders)
        LM_DenseSchur,                                             // Levenberg-Marquardt with dense Schur solver (exact linear solver best suited for camera BA)
        LM_DenseSchur_NonMonotonic,                                // Levenberg-Marquardt with dense Schur solver (exact linear solver best suited for camera BA) allowing a few non-montonic steps during iterations (allows to jump over boulders)
        LM_IterativeSchur_PreconJacobi,                            // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Jacobi matrix.
        LM_IterativeSchur_Implicit_PreconSchurJacobi,              // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Schur Complement S. Implicit evaluation (S*x) of the Schur Complement S and vector x: Best suited for large problems.
        LM_IterativeSchur_Explicit_PreconSchurJacobi,              // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Schur Complement S. Explicit evaluation (S*x) of the Schur Complement S and vector x: Best suited for small problems as S is fully constructed in memory.
        LM_IterativeSchur_PreconJacobi_NonMonotonic,      // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Jacobi matrix and allowing a few non-montonic steps during iterations (allows to jump over boulders).
        LM_IterativeSchur_Implicit_PreconSchurJacobi_NonMonotonic, // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Schur Complement S and allowing a few non-montonic steps during iterations (allows to jump over boulders). Implicit evaluation (S*x) of the Schur Complement S and vector x: Best suited for large problems.
        LM_IterativeSchur_Explicit_PreconSchurJacobi_NonMonotonic, // Levenberg-Marquardt with iterative Schur solver (inexact/approximate linear solver best suited for camera BA) that uses a preconditioner on the Schur Complement S and allowing a few non-montonic steps during iterations (allows to jump over boulders). Explicit evaluation (S*x) of the Schur Complement S and vector x: Best suited for small problems as S is fully constructed in memory.
        Dogleg_SparseNormalChol,                                   // Dogleg algorithm with sparse normal Choleski solver (exact linear solver)
        Dogleg_SparseNormalChol_NonMonotonic,                      // Dogleg algorithm with sparse normal Choleski solver (exact linear solver) allowing a few non-montonic steps during iterations (allows to jump over boulders)
        Dogleg_DenseSchur,                                         // Dogleg algorithm with dense Schur solver (exact linear solver best suited for camera BA)
        Dogleg_DenseSchur_NonMonotonic                             // Dogleg algorithm with dense Schur solver (exact linear solver best suited for camera BA) allowing a few non-montonic steps during iterations (allows to jump over boulders)
    };
}
