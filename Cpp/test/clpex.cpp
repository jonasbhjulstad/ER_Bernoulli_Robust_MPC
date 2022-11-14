#include <coin/ClpSimplex.hpp>

void solve()
{
    std::vector<double> obj = { 1.0, 2.0, 3.0 };
    //s.t x + 2y + 3z <= 4
    CoinPackedMatrix M;
    M.setDimensions(0, 3);
    M.appendRow(3, new int[3]{ 0, 1, 2 }, new double[3]{ 1.0, 1.0, 1.0 });
    std::vector<double> rhs = { 4.0 };
    std::vector<char> sense = { 'L' };
    ClpSimplex model;
    model.loadProblem(M, NULL, NULL, &obj[0], rhs.data(), NULL, NULL);
    model.setLogLevel(0);
    model.primal();
    std::cout << "Objective value: " << model.objectiveValue() << std::endl;
    std::cout << "x: " << model.primalColumnSolution()[0] << std::endl;
    std::cout << "y: " << model.primalColumnSolution()[1] << std::endl;
    std::cout << "z: " << model.primalColumnSolution()[2] << std::endl;
    //check if solve was success
    if (model.status() != 0)
    {
        std::cout << "Error: " << model.status() << std::endl;
    }

}

int main()
{
   
   solve();

}