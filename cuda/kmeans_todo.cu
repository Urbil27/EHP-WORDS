
  // EXEKUTATZEKO: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]    // numwords: matrize txikiekin probak egiteko 

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>



#define VOCAB_SIZE  10000   	// Hitz kopuru maximoa -- Maximo num. de palabras
#define EMB_SIZE    50  	// Embedding-en kopurua hitzeko -- Nº de embedding-s por palabra
#define TAM         25		// Hiztegiko hitzen tamaina maximoa -- Tamaño maximo del diccionario
#define MAX_ITER    1000    	// konbergentzia: iterazio kopuru maximoa -- Convergencia: num maximo de iteraciones
#define K	    20 		// kluster kopurua -- numero de clusters
#define DELTA       0.5		// konbergentzia (cvi) -- convergencia (cvi)
#define NUMCLUSTERSMAX 100	// cluster kopuru maximoa -- numero máximo de clusters

struct clusterinfo	 // clusterrei buruzko informazioa -- informacion de los clusters
{
   int  elements[VOCAB_SIZE]; 	// osagaiak -- elementos
   int  number;       		// osagai kopurua -- número de elementos
};


// Bi bektoreen arteko biderketa eskalarra kalkulatzeko funtzioa
// Función para calcular el producto escalar entre dos vectores
__device__ float d_dot_product(float* a, float* b, int dim) {
    float result = 0;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

float dot_product(float* a, float* b, int dim) {
    float result = 0;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Bi bektoreen arteko norma (magnitudea) kalkulatzeko funtzioa
// Función para calcular la norma (magnitud) de un vector
__device__ float d_magnitude(float* vec, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

float magnitude(float* vec, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}
// Bi bektoreen arteko kosinu antzekotasuna kalkulatzeko funtzioa
// Función para calcular la similitud coseno entre dos vectores
__device__ float d_cosine_similarity(float* vec1, float* vec2, int dim) {
    float mag1, mag2;
    
    mag1 = d_magnitude(vec1, dim);
    mag2 = d_magnitude(vec2, dim);
    if (mag1 == 0 || mag2 == 0) return 0; // Bektoreren bat 0 bada -- Si alguno de los vectores es nulo: cosine_similarity = 0
    else return d_dot_product(vec1, vec2, dim) / (mag1 * mag2);
}

float cosine_similarity(float* vec1, float* vec2, int dim) {
    float mag1, mag2;
    
    mag1 = magnitude(vec1, dim);
    mag2 = magnitude(vec2, dim);
    if (mag1 == 0 || mag2 == 0) return 0; // Bektoreren bat 0 bada -- Si alguno de los vectores es nulo: cosine_similarity = 0
    else return dot_product(vec1, vec2, dim) / (mag1 * mag2);
}
// Distantzia euklidearra: bi hitzen kenketa ber bi, eta atera erro karratua
// Distancia euclidea: raiz cuadrada de la resta de dos palabras elevada al cuadrado
// Adi: double
double word_distance (float *word1, float *word2)
{
  /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
    ****************************************************************************************/
  float bat_partziala = 0;
  int i ;
  for (i=0;i<EMB_SIZE; i++)
  {
    float dif = word1[i]-word2[i];
    bat_partziala+=dif*dif;
  }
  return sqrt(bat_partziala);
}

// Zentroideen hasierako balioak ausaz -- Inicializar centroides aleatoriamente
////////////////////////////
//Funtzio hau paralelizatu//
////////////////////////////
void initialize_centroids(float *words, float *centroids, int n, int numclusters, int dim) {
    int i, j, random_index;
    for (i = 0; i < numclusters; i++) {
        random_index = rand() % n;
        for (j = 0; j < dim; j++) {
            centroids[i*dim+j] = words[random_index*dim+j];
        }
    }
}

// Zentroideak eguneratu -- Actualizar centroides
////////////////////////////
//Funtzio hau paralelizatu//  ////OSO INPORRTANTEA DENBORAK OPTIMIZATZEKO
////////////////////////////
__global__ void update_centroids_fase1(float *centroids, int numclusters, int dim, int *cluster_sizes) {
    
    int i, j;
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < numclusters; i+=stride) {
        cluster_sizes[i]=0;
        for (int j = 0; j < dim; j++) {
            centroids[i*dim+j] = 0.0; // Zentroideak berrasieratu -- Reinicia los centroides
        }
    }
  }

  __global__ void update_centroids_fase2(float *words, float *centroids, int *wordcent, int numwords, int dim, int *cluster_sizes) {
    int i, j,cluster;
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (i = id; i < numwords; i+=stride) {
        cluster = wordcent[i];
        atomicAdd(&cluster_sizes[cluster], 1);
        for (j = 0; j < dim; j++) {
          atomicAdd(&centroids[cluster*dim+j], words[i*dim+j]);
        }
    }
  }
__global__ void update_centroids_fase3(float *centroids, int numclusters, int dim, int *cluster_sizes){
  int i, j;
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  
  for (i = id; i < numclusters; i+=stride) {
        if (cluster_sizes[i] > 0) {
            for (j = 0; j < dim; j++) {
                centroids[i * dim + j] = centroids[i * dim + j] / cluster_sizes[i];
            }
        }
    }
}

// K-Means funtzio nagusia -- Función principal de K-Means
__global__ void k_means_calculate(float *words, int numwords, int dim, int numclusters, int *wordcent, float *centroids, int *changed) 
{  
/****************************************************************************************    
           OSATZEKO - PARA COMPLETAR
           - Hitz bakoitzari cluster gertukoena esleitu cosine_similarity funtzioan oinarrituta
           - Asignar cada palabra al cluster más cercano basandose en la función cosine_similarity       
****************************************************************************************/
////////////////////////////
//Funtzio hau paralelizatu//
////////////////////////////


int id = (blockIdx.x * blockDim.x) + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for(int i=id;i<numwords;i+=stride) //Hitzak iteratu
{
  int gert_cluster_idx = -1;
  float gert_cluster_dist =-2.0f;
  for(int j=0; j<numclusters; j++) //Clusterrak iteratu
  {
    float uneko_dist = d_cosine_similarity(words+i*dim,centroids+j*dim,dim);
    if(uneko_dist>gert_cluster_dist || j == 0) //uneko distantzia txikiagoa izango da similaritatea handiagoa bada
    {  
    //balioak eguneratu
      gert_cluster_dist = uneko_dist;
      gert_cluster_idx = j;
    }
  }
   if (wordcent[i] != gert_cluster_idx) {
            atomicOr(changed,1) ;  
        }
  //Aurkitu dugu gertuen dagoen clusterra, sartu bektorean
  wordcent[i] = gert_cluster_idx; 
}
}

__device__ double d_cluster_homogeneity(float *words, struct clusterinfo *members, int wd_idx, int numclusters, int number) //zertarako nahi ditut numclusters eta number?
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       Kideen arteko distantzien batezbestekoa - Media de las distancias entre los elementos del cluster
       Cluster bakoitzean, hitz bikote guztien arteko distantziak - En cada cluster, las distancias entre todos los pares de elementos
       Adi, i-j neurtuta, ez da gero j-i neurtu behar  / Ojo, una vez calculado el par i-j no hay que calcular el j-i
    ****************************************************************************************/
  int j, n = members[wd_idx].number, kontatutako_bikoteak = 0;
  double dist_bb=0.0, dist_batura=0.0;
  for(int i = 0; i<n; i++)
  {
    for(int j = i+1; j<n; j++)
    {
     
        int word_idx1 = members[wd_idx].elements[i];
        int word_idx2 = members[wd_idx].elements[j];
        dist_batura+=1.0f-d_cosine_similarity(
          &words[word_idx1 * EMB_SIZE],   // 1. hitzeko embeddingaren punteroa
          &words[word_idx2 * EMB_SIZE],    // 2. hitzeko embeddingaren punteroa
          EMB_SIZE
        );
        kontatutako_bikoteak++;
    }

  }
  if (kontatutako_bikoteak > 0) {
        dist_bb=dist_batura / kontatutako_bikoteak;
    }
  return dist_bb;

}

double cluster_homogeneity(float *words, struct clusterinfo *members, int wd_idx, int numclusters, int number) //zertarako nahi ditut numclusters eta number?
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       Kideen arteko distantzien batezbestekoa - Media de las distancias entre los elementos del cluster
       Cluster bakoitzean, hitz bikote guztien arteko distantziak - En cada cluster, las distancias entre todos los pares de elementos
       Adi, i-j neurtuta, ez da gero j-i neurtu behar  / Ojo, una vez calculado el par i-j no hay que calcular el j-i
    ****************************************************************************************/
  int j, n = members[wd_idx].number, kontatutako_bikoteak = 0;
  double dist_bb=0.0, dist_batura=0.0;
  for(int i = 0; i<n; i++)
  {
    for(int j = i+1; j<n; j++)
    {
     
        int word_idx1 = members[wd_idx].elements[i];
        int word_idx2 = members[wd_idx].elements[j];
        dist_batura+=1.0f-cosine_similarity(
          &words[word_idx1 * EMB_SIZE],   // 1. hitzeko embeddingaren punteroa
          &words[word_idx2 * EMB_SIZE],    // 2. hitzeko embeddingaren punteroa
          EMB_SIZE
        );
        kontatutako_bikoteak++;
    }

  }
  if (kontatutako_bikoteak > 0) {
        dist_bb=dist_batura / kontatutako_bikoteak;
    }
  return dist_bb;

}

double centroid_homogeneity(float *centroids, int cl_idx, int numclusters)
{
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
    ****************************************************************************************/
    int i,j; 
    double dist_bb, dist_batura=0.0;
    for(int i = 0; i<numclusters;i++)
    {
    
        if (cl_idx!=i)
        {
          dist_batura+=1.0f-cosine_similarity(
            &centroids[cl_idx * EMB_SIZE],
            &centroids[i * EMB_SIZE],
            EMB_SIZE);
        }
      
    }
    dist_bb=dist_batura/(numclusters-1);
    return dist_bb;
}


double validation (float *words, struct clusterinfo *members, float *centroids, int numclusters)
{
  int     i, j, k, number, word1, word2;
  float   cent_homog[numclusters];
  double  disbat, max, cvi;
  float   clust_homog[numclusters];	// multzo bakoitzeko trinkotasuna -- homogeneidad de cada cluster

  // Kalkulatu clusterren trinkotasuna -- Calcular la homogeneidad de los clusters
  // Cluster bakoitzean, hitz bikote guztien arteko distantzien batezbestekoa. Adi, i - j neurtuta, ez da gero j - i neurtu behar
  // En cada cluster las distancias entre todos los pares de palabras. Ojo, una vez calculado i - j, no hay que calcular el j - i
  //Intra cluster distantziak kalkulatu.
 for (i=0; i<numclusters; i++)
  {
    disbat = 0.0;
    number = members[i].number; 
    if (number > 1)     // min 2 members in the cluster
    {
       disbat = cluster_homogeneity(words, members, i, numclusters, number);
       clust_homog[i] = disbat/(number*(number-1)/2);	// zati bikote kopurua -- div num de parejas
    }
    else clust_homog[i] = 0;


  // Kalkulatu zentroideen trinkotasuna -- Calcular la homogeneidad de los centroides
  // clusterreko zentroidetik gainerako zentroideetarako batez besteko distantzia 
  // dist. media del centroide del cluster al resto de centroides
  
    disbat = centroid_homogeneity(centroids, i, numclusters);
    cent_homog[i] = disbat/ (numclusters-1);	// 5 multzo badira, 4 distantzia batu dira -- si son 5 clusters, se han sumado 4 dist.
  }
  
  // cvi index
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
      fmaxf: max of 2 floats --> maximoa kalkulatzeko -- para calcular el máximo
    ****************************************************************************************/
    float max_intra = -1e20f;
    double cvi_batukaria = 0.0f;
    for (i=0; i<numclusters;i++)
    {
      cvi_batukaria+=(cent_homog[i]-clust_homog[i])/fmaxf(cent_homog[i],clust_homog[i]);
    }
    
    cvi = cvi_batukaria/numclusters; // zenbat eta txikiagoa, orduan eta hobea da clustering-a
  return (cvi);
}


int main(int argc, char *argv[]) 
{
    int		i, j, numwords, k, iter, changed = 0, *d_changed, end_classif;
    int		cluster, zenb, numclusters = 20;
    double  	cvi, cvi_zaharra, dif;
    float 	*words, *d_words;
    FILE    	*f1, *f2, *f3;
    char 	**hiztegia;  
    int     	*wordcent, *d_wordcent;

    struct clusterinfo  members[NUMCLUSTERSMAX];

    struct timespec  t0, t1;
    double tej;
 

   if (argc < 4) {
     printf("\nCall: kmeans embeddings.dat dictionary.dat myclusters.dat [numwords]\n\n");
     printf("\t(in) embeddings.dat and dictionary.dat\n");
     printf("\t(out) myclusters.dat\n");
     printf("\t(numwords optional) prozesatu nahi den hitz kopurua -- num de palabras a procesar\n\n");
     exit (-1);;
   }  
   
  // Irakurri datuak sarrea-fitxategietatik -- Leer los datos de los ficheros de entrada
  // =================================================================================== 

  f1 = fopen (argv[1], "r");
  if (f1 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[1]);
    exit (-1);
  }

  f2 = fopen (argv[2], "r");
  if (f2 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[2]);
    exit (-1);
  }
  
  fscanf (f1, "%d", &numwords);	       
  if (argc == 5) numwords = atoi (argv[4]);  
  printf ("numwords = %d\n", numwords);

  words = (float*)malloc (numwords*EMB_SIZE*sizeof(float));
  hiztegia = (char**)malloc (numwords*sizeof(char*));
 
  for (i=0; i<numwords;i++){
   hiztegia[i] = (char*)malloc(TAM*sizeof(char));
  }
  
  for (i=0; i<numwords; i++) {
   fscanf (f2, "%s", hiztegia[i]);
   for (j=0; j<EMB_SIZE; j++) {
    fscanf (f1, "%f", &(words[i*EMB_SIZE+j]));
   }
  }
  printf ("Embeddingak eta hiztegia irakurrita -- Embeddings y dicionario leidos\n");

  wordcent = (int *)malloc(numwords * sizeof(int));

 
  for (int i = 0; i < numwords; i++) wordcent[i] = -1;

  k = NUMCLUSTERSMAX;   // hasierako kluster kopurua (20) -- numero de clusters inicial
  end_classif = 0; 
  cvi_zaharra = -1;
  
  float *centroids = (float *)malloc(k * EMB_SIZE * sizeof(float));
  float *d_centroids;
  int *cluster_sizes = (int *)calloc(k, sizeof(int));
  int *d_cluster_sizes;
  int bltam = 256;
  int blkop = (numwords + bltam - 1) / bltam;
  //Memoria dinamikoki erreserbatu txartelean
  cudaMalloc(&d_words, numwords*EMB_SIZE*sizeof(float));
  cudaMalloc(&d_wordcent, numwords * sizeof(int));
  cudaMalloc(&d_centroids,k * EMB_SIZE * sizeof(float));
  cudaMalloc(&d_changed,sizeof(int));
  cudaMalloc(&d_cluster_sizes, k*sizeof(int));

  //Datuak kopiatu txartelera
  cudaMemcpy(d_words,words,numwords*EMB_SIZE*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_wordcent,wordcent,numwords * sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cluster_sizes,cluster_sizes,k*sizeof(int),cudaMemcpyHostToDevice);





/******************************************************************/
  // A. kmeans kalkulatu -- Calcular kmeans
  // =========================================================
  printf("K_means\n");
  clock_gettime (CLOCK_REALTIME, &t0);
  
  while (numclusters < NUMCLUSTERSMAX && end_classif == 0)
  {
    initialize_centroids(words, centroids, numwords, numclusters, EMB_SIZE);
    cudaMemcpy(d_centroids, centroids, numclusters * EMB_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (iter = 0; iter < MAX_ITER; iter++) {
      changed = 0;
      cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
       deitu k_means_calculate funtzioari -- llamar a la función k_means_calculate
    ****************************************************************************************/
    k_means_calculate  <<< blkop, bltam >>> (d_words, numwords,EMB_SIZE,numclusters,d_wordcent,d_centroids,d_changed);

   
    cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    
      if (changed==0) break; // Aldaketarik ez bada egon, atera -- Si no hay cambios, salir
      cudaMemset(d_cluster_sizes, 0, numclusters * sizeof(int));

      update_centroids_fase1 <<<blkop,bltam>>> (d_centroids, numclusters, EMB_SIZE, d_cluster_sizes);
      update_centroids_fase2 <<<blkop,bltam>>> (d_words, d_centroids, d_wordcent, numwords, EMB_SIZE, d_cluster_sizes);
      update_centroids_fase3 <<<blkop,bltam>>> (d_centroids, numclusters, EMB_SIZE, d_cluster_sizes);

      

    }  
    cudaMemcpy(words,d_words,numwords*EMB_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(wordcent,d_wordcent,numwords * sizeof(int),cudaMemcpyDeviceToHost);
      cudaMemcpy(centroids,d_centroids,k * EMB_SIZE * sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(centroids, d_centroids, numclusters * EMB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(cluster_sizes,d_cluster_sizes,k*sizeof(int),cudaMemcpyDeviceToHost);


  // B. Sailkatzearen "kalitatea" -- "Calidad" del cluster
  // =====================================================
    printf("Kalitatea -- Calidad\n");   
    for (i=0; i<numclusters; i++)  members[i].number = 0;

    // cluster bakoitzeko hitzak (osagaiak) eta kopurua -- palabras de cada clusters y cuantas son
    for (i=0; i<numwords; i++)  {
      cluster = wordcent[i];
      zenb = members[cluster].number;
      members[cluster].elements[zenb] = i;	// clusterreko hitza -- palabra del cluster
      members[cluster].number ++; 
    }
    
    /****************************************************************************************
      OSATZEKO - PARA COMPLETAR
        cvi = validation (OSATZEKO - PARA COMPLETAR);
   	if (cvi appropriate) end classification;
        else  continue classification;	
    ****************************************************************************************/
    cvi = validation(words,members,centroids,numclusters);
    // Begiratu ea konbergitu duesn
   

    if (cvi_zaharra != -1) { //lehenengo iterazioa ekiditu ez dugulako ezer cvi_zaharrean
        dif = fabs(cvi - cvi_zaharra);
        if (dif < DELTA) {
            end_classif = 1;  // ¡CONVERGENCIA ALCANZADA!
        }
    }
    cvi_zaharra = cvi;
    
    if (end_classif == 0) {
        numclusters+=10;  // Incrementar para siguiente iteración
    }
  }

  clock_gettime (CLOCK_REALTIME, &t1);
/******************************************************************/
  printf("cvi = %f eta numclusters = %d\n", cvi, numclusters);
  for (i=0; i<numclusters; i++)
  printf ("%d. cluster, %d words \n", i, cluster_sizes[i]);

  tej = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / (double)1e9;
  printf("\n Tej. (serie) = %1.3f ms\n\n", tej*1000);

// Idatzi clusterrak fitxategietan -- Escribir los clusters en el fichero
  f3 = fopen (argv[3], "w");
  if (f3 == NULL) {
    printf ("Errorea %s fitxategia irekitzean -- Error abriendo fichero\n", argv[3]);
    exit (-1);
  }

  for (i=0; i<numwords; i++)
     fprintf (f3, "%s \t\t -> %d cluster\n", hiztegia[i], wordcent[i]);
   printf ("clusters written\n");

  fclose (f1);
  fclose (f2);
  fclose (f3);


  //Txartelean memoria askatu

  cudaFree(d_words);
  cudaFree(d_wordcent);
  cudaFree(d_centroids);
  cudaFree(d_changed);

  free(words);
  for (i=0; i<numwords;i++) free (hiztegia[i]);
  free(hiztegia); 
  free(cluster_sizes);
  free(centroids);
  return 0;
}

