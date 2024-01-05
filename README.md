# CUDA Matrix Multiplication
Multiplication matrix by using CUDA <br>

Each element of the resulting matrix in the operation of multiplication is calculated regardless of the rest, which makes it possible to paralle this operation. In this work, CUDA technology was used to parallelize calculations on the graphic processor. Each CUDA block in this work contains 16 on 16 processes, each of which has cleared one element of the resulting matrix by multiplying vectors. To calculate the acceleration, consistent time is measured on a regular processor and on a graphic one. In addition, a check of the correctness of multiplication is being verified.

To compile use ```nvcc kernel.cu kernel```.

Каждый элемент результирующей матрицы в операции умножения вычисляется независимо от остальных, что дает возможность распаралеллить эту операцию. В этой ЛР была использована технология CUDA для распараллеливания вычислений на графическом процессоре. Каждый CUDA блок в этой ЛР вмещает в себя 16 на 16 процессов, каждый из которых вычисялет один эелемент результирующей матрицы путем умножения векторов. Для вычисления ускорения производится замер времени последовательного исполнения на обычном процессоре и на графическом. Кроме этого, выпоняется проверка корректности умножения.

## Экспериментальные результаты
Эксперименты производились с использованием процессора AMD Ryzen 5 2600 и графического ускорителя NVIDIA GeForce RTX 2060.
| Размер матриц | Время последовательного выполнения, ms  | Время паралелльного выполнения, ms  | Ускорение |
| :-----------: |:---------------------------------------:| :----------------------------------:| :-------: |
| 160           |    13                                   |   323                               |      0.04 |
| 320           |    102                                  |   316                               |      0.32 |
| 640           |    810                                  |   363                               |      2.23 |
| 1280          |    8466                                 |   401                               |      21.11|
| 1600          |    18556                                |   560                               |      33.13|

2020.