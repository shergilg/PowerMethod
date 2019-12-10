"""
    Author: Gurbaksh Shergill
    Student Number: 7779192
    Course: MATH2740
    Assignment: Assignment 5
    Instructor: Dr. Michael Szestopalow

"""
# import statements
import numpy as np
from numpy import linalg as la

"""
    original_A = matrix A given in the question
    iterative_A = matrix A(Xi) where i is the number of iterations that have been performed on A using the initial X0
                    so Xi is X1,X2,X3,X4,.......
    basically iterative_A is i'th iteration of A 



"""

A_problem1 = np.array([[6, 5], [1, 2]])  # Matrix A from problem 1
X0_problem1 = np.array([0, 1])  # X0 from problem 1
A1 = A_problem1.__matmul__(X0_problem1)  # Iteration 0 for problem 1

A_problem2 = np.array([[6, -3], [-3, 6]])  # Matrix A from problem 2
X0_problem2 = np.array([1, 1])  # X0 from problem 2
A2 = A_problem2.__matmul__(X0_problem2)  # Iteration 0 for problem 2

iterations = 5  # number of iterations only from 1 to 5(so far)

# I wrote my own function for the power method
""" Parameters: A(matrix), X0(matrix)
    this function takes 2 parameters(A and X0) and outputs 
"""


# calculates newX for next Iteration
def get_newX(original_A, iterative_A, *args):
    norm = la.norm(iterative_A)  # calculation of the norm
    newX = original_A * (1 / norm)  # calculation of the new Xi (Where i is 1,2,3,4,.......)
    return newX


""" Parameters: original matrix A, i'th Iteration of A
    this function takes 2 parameters(original_A and iterative_A) and outputs i'th Iteration of A
"""


def power_method(original_A, A, *args):
    newX = get_newX(original_A, A)  # Given the original A and the iterative A, find the X value for the new iteration
    next_iteration = A.__matmul__(newX)  # multiplication of A and the Xi
    # norm = la.norm(A)
    # print("AX0 = ", A, " and norm = ", norm)
    # print(A.shape)
    # newX = A * (1 / norm)
    # print("X1 = ", A)
    return next_iteration


"""
 This function calculates the eigenValue and the eigenVector using the built in functions

"""


def calculate_eigenvalue(A, *args):
    eigvals, eigenvec = la.eig(A)
    print("Using det(A-λI) to find the eigenvalues we get: ")
    for i in eigvals:
        print(int(i))

    print("and the corresponding eigenvector is ",
          eigenvec[1])


"""
perform_iteration takes in the (Original A, Iterative A, number of Iterations) as the parameters and Outputs 
i'th iteration of A
"""


def perform_iterations(A, iterative_A, iterations, *args):
    if iterations is 1:
        print("\nAnswer using power method is ", power_method(A, iterative_A))
    elif iterations is 2:
        print("\nAnswer using power method is ", power_method(A, power_method(A, iterative_A)))
    elif iterations is 3:
        print("\nAnswer using power method is ", power_method(A, power_method(A, power_method(A, iterative_A))))
    elif iterations is 4:
        print("\nAnswer using power method is ", power_method(A, power_method(A, power_method(A, power_method(
            A, iterative_A)))))
    elif iterations is 5:
        print("\nAnswer using power method is ",
              power_method(A, power_method(A, power_method(A, power_method(A, power_method(
                  A, iterative_A))))))
    else:
        print("\nSorry! I am using handwritten method for iterations at the moment. WILL UPDATE WHEN I FIND THE ACTUAL "
              "SOLUTION(Only works with iterations 1 to 5 right now)")


# Calling the methods to solve problem 1
print("Problem 1 Solution:")
calculate_eigenvalue(A_problem1)
perform_iterations(A_problem1, A1, iterations)

# Calling the methods to solve problem 2
print("\nProblem 2 Solution:")
calculate_eigenvalue(A_problem2)
perform_iterations(A_problem2, A2, iterations)

"""
ANSWER FOR PROBLEM 2
"""
# The vector X0 must be pointing in the same direction. If it is perpendicular or pointing in the
# direction, it would take far too many iterations



# End of Processing
