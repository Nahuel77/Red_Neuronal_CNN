Red Neuronal Convolucional (CNN) que detecta números escritos a mano.

Anteriormente habia creado una red MLP que realizaba la misma tarea.

Ahora hare el paso de MLP a CNN con el proposito de mejorar la presición

Arquitectura típica de una CNN

Entrada (imagen) 
       ↓
Convolución + ReLU
       ↓
Pooling
       ↓
Convolución + ReLU
       ↓
Pooling
       ↓
Flatten
       ↓
Fully Connected
       ↓
Softmax (clasificación)

<h3>Si bien en MLP iniciabamos pesos al azar y los ajustabamos con retropropagación. Aqui lo que se iniciará y ajustaran seran los Kernels de las convoluciones.</h3>