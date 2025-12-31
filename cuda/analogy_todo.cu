  // KONPILATZEKO - PARA COMPILAR: (C: -lm) (CUDA: -arch=sm_61)
  // EXEC: analogy embeddings.dat dictionary.dat 
  // Ej., king – man + woman = queen

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 10000     // Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE 50  	     // Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM 25		     // Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario


// Hitz baten indizea kalkulatzeko funtzioa
// Función para calcular el indice de una palabra 
int word2ind(char* word, char** dictionary, int numwords) {
    for (int i = 0; i < numwords; i++) {
        if (strcmp(word, dictionary[i]) == 0) {
            return i;
        }
    }
    return -1;  // if the word is not found
}

// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
__device__ double dot_product(float* a, float* b, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
__device__ float magnitude(float* vec, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
__device__ float cosine_similarity(float* vec1, float* vec2, int size) {
    float mag1, mag2;
    
    mag1 = magnitude(vec1, size);
    mag2 = magnitude(vec2, size);
    return dot_product(vec1, vec2, size) / (mag1 * mag2);
}

//Lehen kernela
__global__ void perform_analogy(float *words, int idx1, int idx2, int idx3, float *result_vector) {
  int id;
  id = (blockIdx.x * blockDim.x) + threadIdx.x;
	result_vector[id]= words[idx1*EMB_SIZE+id]-words[idx2*EMB_SIZE+id]+words[idx3*EMB_SIZE+id];
 
 } 

//Bigarren Kernela
__global__ void compute_similarities(float *result_vector, float *words, int numwords, int idx1, int idx2, int idx3, float *similarities) {
  int i;
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(i=id;i<numwords;i+=stride)
  {
    if (i == idx1 || i == idx2 || i == idx3){ 
      similarities[i] = -1.0f;
      continue;
    }
    similarities[i] = cosine_similarity(result_vector,&words[i*EMB_SIZE],EMB_SIZE);
}
}

int main(int argc, char *argv[]) 
{
    int		i, j, numwords, idx1, idx2, idx3, closest_word_idx;
    float	max_similarity;
    float 	*words, *d_words, *result_vector, *d_result_vector, *sim_cosine;
    FILE    	*f1, *f2;
    char 	**dictionary;  
    char	target_word1[TAM], target_word2[TAM], target_word3[TAM];
        
    struct timespec  t0, t1;
    double tej;
    int blkop = 1, bltam = EMB_SIZE;
  

   if (argc < 3) {
     printf("Deia: analogia embedding_fitx hiztegi_fitx\n");
     exit (-1);
   }  
   
   

  f1 = fopen (argv[1], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean\n", argv[1]);
    exit (-1);
  }

  f2 = fopen (argv[2], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean\n", argv[2]);
    exit (-1);
  }
  
 
  fscanf (f1, "%d", &numwords);	  
  if (argc == 4) numwords = atoi (argv[3]);  
 printf ("numwords = %d\n", numwords);
  
  words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
  cudaMalloc(&d_words, numwords*EMB_SIZE*sizeof(float));
  dictionary = (char**)malloc (numwords*sizeof(char*));
  for (i=0; i<numwords;i++){
   dictionary[i] = (char*)malloc(TAM*sizeof(char));
  }
  sim_cosine = (float*)malloc (numwords*sizeof(float));
  result_vector = (float*)malloc (EMB_SIZE*sizeof(float));
  cudaMalloc(&d_result_vector, (EMB_SIZE*sizeof(float)));
    float *similarities; // Puntero para la memoria de la CPU
    similarities = (float*)malloc(numwords * sizeof(float));

    float *d_similarities; // Puntero para la memoria de la GPU
    cudaMalloc((void**)&d_similarities, numwords * sizeof(float));
  for (i=0; i<numwords; i++) {
   fscanf (f2, "%s", dictionary[i]);
   for (j=0; j<EMB_SIZE; j++) {
    fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
   }
  }
  printf("Sartu analogoak diren bi hitzak eta analogia bilatu nahi diozun hitza: \n");
  printf("Introduce las dos palabras analogas y la palabra a la que le quieres buscar la analogia: \n");
  scanf ("%s %s %s",target_word1, target_word2, target_word3);
  
  //Hitzetik indizera esleitu
  idx1 = word2ind(target_word1,dictionary, numwords);
  idx2 = word2ind(target_word2,dictionary, numwords);
  idx3 = word2ind(target_word3,dictionary, numwords);


  if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
     printf("Errorea: Ez daude hitz guztiak hiztegian / No se encontraron todas las palabras en el vocabulario.\n");
     return -1;
  }
   
  clock_gettime (CLOCK_REALTIME, &t0);

  cudaMemcpy(d_words, words ,numwords*EMB_SIZE*sizeof(float),cudaMemcpyHostToDevice);
   
  perform_analogy <<< blkop, bltam>>> (d_words, idx1, idx2, idx3, d_result_vector);

  cudaMemcpy(result_vector,d_result_vector,EMB_SIZE*sizeof(float) , cudaMemcpyDeviceToHost);
  
  int bltam2 = 256;
  int blkop2 = (numwords + bltam2 - 1) / bltam2;
  compute_similarities <<<blkop2, bltam2>>> (d_result_vector,d_words,numwords,idx1,idx2,idx3, d_similarities);
    
  cudaDeviceSynchronize();
    
  cudaMemcpy(similarities, d_similarities, numwords * sizeof(float), cudaMemcpyDeviceToHost);

  max_similarity = -1.0f;
  closest_word_idx = -1; // Esta es la variable que antes intentabas pasar al kernel

  for (int i = 0; i < numwords; i++) {
      if (similarities[i] > max_similarity) {
          max_similarity = similarities[i];
          closest_word_idx = i;
      }
  }

  clock_gettime (CLOCK_REALTIME, &t1);   
   
    if (closest_word_idx != -1) {
        printf("\nClosest_word: %s (%d), sim = %f \n", dictionary[closest_word_idx],closest_word_idx, max_similarity);
    } else printf("No close word found.\n");
 
  
  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (cuda) = %1.3f ms\n\n", tej*1000);

  fclose (f1);
  fclose (f2);
  
  free(words);
  free(sim_cosine);
  free(result_vector);
  for (i=0; i<numwords;i++) free (dictionary[i]);
  free(dictionary); 

  return 0;
}

