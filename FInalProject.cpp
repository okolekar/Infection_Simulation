#include <iostream>
#include <random>
#include "mpi.h"
#include <unistd.h>                                                                                                      // For usleep function

const static float q           = 0.3;
const static int immune_time   = 3;
const static int R             = 20;                                                                                     //Total number of rows 
const static int c             = 20;                                                                                     //Total number of columns

void ResetRecoverImmune(float (*Mx)[c], int r);
void Infect(float (*Mx)[c], int r, int m,int n, int rank);
void RowCorrector(float (*Mx)[c], float (*Mcr), int cRow);
void print(float (*Mp)[c], int r);


int main(int argc, char **argv){

    int rank, size;
    MPI_Init(&argc, &argv);
    int r;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float Mshare[c];
    MPI_Status status;
    //--------------------------------------------------------------------------------------------------------------------------------------//

    for(int i=0;i<size;i++){
        if(rank==size-1){
            r = R/size + R%size;
	    //std::cout<<"The r is "<< r<<" for the rank " << rank << std::endl;
        }
        else{
            r = R/size;
        }
    }
    float M1[r][c];
    float M2[r][c];

    //------------------------------------------------Initilizing the matrices---------------------------------------------------------------//
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            M1[i][j]=0;
            M2[i][j]=0;
        }
    }
    //--------------------------------------------------------------------------------------------------------------------------------------//

    std::random_device rd;                                                                                              // Get a random seed from the device
    std::mt19937 gen(rd());                                                                                             // Initialize the Mersenne Twister random number generator
    std::uniform_int_distribution<> distribution(0, r-1);                                                                 // Define the distribution (0 to 7 inclusive)
    std::uniform_real_distribution<float> infection_Probability_Distribution(0.0f, 1.0f);                               
                                                                                                                        
    int random_infected = distribution(gen);                                                                            // Generate a random number
    float resilience;
    float recovery_probability;

    //-----------------------------------------------------------------------------------------------------------------------------------//
    Infect(M1,r, random_infected,random_infected,rank);

    //####################################################Debugging###################################################//
    /*if (rank == 0) {
    std::cout << "Testing rank 0 just after Infection " << std::endl;
    print(M1, r);
    }*/
    //-----------------------------------------------------------------------------------------------------------------------------------//

    if(size>1){
        //####################################################Starting the send block###################################################//
        if(rank==0){
            MPI_Send(&M1[r-1][0], c,MPI_FLOAT,1,112,MPI_COMM_WORLD);                                                   //Rank 0 sends the last row to the rank 1
        }
        else if(rank==size-1){                                          
            MPI_Send(&M1[0][0], c,MPI_FLOAT,size-2,112,MPI_COMM_WORLD);                                                //Last rank sends the first row to the second last rank
        }
        else {
            MPI_Send(&M1[r-1][0], c,MPI_FLOAT,rank+1,112,MPI_COMM_WORLD);                                              //Inbetween ranks sends first row to the previous rank 
            MPI_Send(&M1[0][0], c,MPI_FLOAT,rank-1,112,MPI_COMM_WORLD);                                                //and the last row to the next rank
        }
        MPI_Barrier(MPI_COMM_WORLD);                                                                                   //Barrier to ensure all ranks have finished sending the rows.

        //##################################################Starting the receive block#################################################//
        if(rank==0){
            MPI_Recv(Mshare, c, MPI_FLOAT, 1, 112, MPI_COMM_WORLD, &status);                                          //Rank 0 receives the first row from the rank 1
            RowCorrector(M1, Mshare, r-1);
        }
        else if(rank==size-1){                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, size-2, 112, MPI_COMM_WORLD, &status);                                     //Last rank receives the last row from the second last rank
            RowCorrector(M1, Mshare, 0);
        }                                                                                                               
        else {                                                                                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, rank+1, 112, MPI_COMM_WORLD, &status);                                    //Inbetween ranks receives last row from the previous rank
            RowCorrector(M1, Mshare, r-1); 
            MPI_Recv(Mshare, c, MPI_FLOAT, rank-1, 112, MPI_COMM_WORLD, &status);                                    //and the first row from the next rank
            RowCorrector(M1, Mshare, 0);
        } 
    }
    
        for (int i = 0; i < r; i++){
                for(int j= 0; j < c; j++){
                            M2[i][j] = M1[i][j];
                                    }
		}
    
//--------------------------------------------------------------------------------------------------------------------------------------//

    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout<<"The rank printing is "<<rank<<std::endl;
    /*for(int rank_nos = 0;rank_nos < size;rank_nos++){
        if(rank != 0 && rank != size-1){
            
        }

    }*/
    //The time settings here
    //Call the reset recover function call
    for(int node=0;node<size;node++){
        if (node == rank){
        std::cout << "The rank printing is " << rank << std::endl;
        print(M1,r);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

void RowCorrector(float (*Mx)[c], float (*Mcr), int cRow){
    for(int i=0;i<c;i++){
        if(Mcr[i]==1){
            if(i==0){                                                                                                   //This if statement takes care of the left corner point
                if(Mx[cRow][i]<1.0){Mx[cRow][i] += 0.1;}                                                                //This if statement takes care of the right corner point
                if(Mx[cRow][i+1]<1.0){Mx[cRow][i+1] += 0.1;}                                                                
            }                                                                                                                       
            else if(i==c-1){                                                                            
                if(Mx[cRow][i]<1.0){Mx[cRow][i] += 0.1;}                                                                
                if(Mx[cRow][i-1]<1.0){Mx[cRow][i-1] += 0.1;}                                                                
            }                                                                                               
            else {                                                                                                      //This else statement takes care of all other points
                if(Mx[cRow][i-1]<1.0){Mx[cRow][i-1] += 0.1;}
                if(Mx[cRow][i]<1.0){Mx[cRow][i] += 0.1;}
                if(Mx[cRow][i+1]<1.0){Mx[cRow][i+1] += 0.1;}
            }
        }
    }
}

void ResetRecoverImmune(float (*Mx)[c], int r){
	float resilience;
	float recovery_probability;
	int flag;
	std::uniform_real_distribution<float> infection_Probability_Distribution(0.0f, 1.0f);
	for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            std::random_device rd;
            std::mt19937 gen(rd());
            recovery_probability = infection_Probability_Distribution(gen);
            if(recovery_probability > q){
                flag = 1;
            }
            else{
                flag = 0;
            }
            if (Mx[i][j] >= 1 && Mx[i][j] < immune_time+1 && flag == 1){
                std::cout << "The value at The indices i and j are as follows " << i << " " << j <<  " " << Mx[i][j] << std::endl;
                //#pragma omp parallel for private(i,j) shared (Mx,i,j)
                for (int k=i-1;k<i+2;k++){
                    for (int l=j-1;l<j+2;l++){
                        if (k==i && l==j){
                            Mx[i][j] = Mx[i][j]+1;                                                                       //Removes the infection flag and sets the value in the cell = the time past, for which it is immune
                        }
                        else{
                            if (Mx[k][l]<1.0 && Mx[k][l]>=0.1){
                                Mx[k][l] = Mx[k][l]-0.1;
                            }
                        }
                    }
                }
            }
            if (Mx[i][j] > 1 && Mx[i][j] < immune_time+1 && flag == 0){                                                   //Increases the Time for already Immune Cell 
                Mx[i][j] = Mx[i][j]+1.0;                                                                                  //Affects to the cells which are not infected but are under immunity time
            }
            if (Mx[i][j] == immune_time+1){
                Mx[i][j] = 0.0;                                                                                          //Wanes the immunity and makes the cell vulnerable for infection. 
            }
            if (Mx[i][j] > immune_time+1){
                std::cout<<"WARNING: the cell value at the position ("<< i << ", "<<j<<") is invalid."<< std::endl;      //Protection against false values.
            }
        }
    }
}


void Infect(float (*Mx)[c], int r, int m,int n, int rank){
    int tr,tc,er,ec;
    std::cout<<"The row and column for the rank "<< rank <<" is r="<<m<<"and c="<<n<<std::endl;
    if(m==0){
        tr = 0;
        er = m + 2;
    } else if(m == r-1){
        tr = r-2;
        er = m+1;
    }else{
        tr = m-1;
        er = m+2;
    }
    

    switch(n){
        case 0:
            tc = 0;
            ec = n + 2;
            break;
        case c-1:
            tc = c-2;
            ec = n+1;
            break;
        default:
            tc = n-1;
            ec = n + 2;
    }

    //#pragma omp parallel for private(i,j) shared (tr,tc,er,ec,Mx)
    for(int i=tr;i<er;i++){
        for(int j=tc;j<ec;j++){
            if (Mx[i][j] < 1 ){
                Mx[i][j] = Mx[i][j] + 0.1;
                //Mx->M[i][j] = 0.2;
                //std::cout<<"I am adding 0.2 at position "<<i<<" "<<j<<std::endl;                                      //For Debugging purpose
            } 
        }
    }
    if(Mx[m][n]<=1){
        Mx[m][n] = 1;
    }
}


void print(float (*Mp)[c], int r){
    float cell_value;
    for (int i =0; i< r;i++){
        for (int j =0; j<c;j++){
            if (Mp[i][j] < 0.001){
                cell_value = 0.0;
                } 
            else {
                cell_value = Mp[i][j];
                }
            std::cout<<"    "<< cell_value <<"    ";
        }
        std::cout<<std::endl;
    }
}
