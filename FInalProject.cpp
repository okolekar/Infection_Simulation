#include <iostream>
#include <random>
#include "mpi.h"
#include <unistd.h>                                                                                                      // For usleep function
#include <algorithm>


const static float q           = 0.3;
const static int immune_time   = 3;
const static int R             = 15;                                                                                     //Total number of rows 
const static int c             = 15;                                                                                     //Total number of columns

void ResetRecoverImmune(float (*Mx)[c], int r, int rank, int size);
void Infect(float (*Mx)[c], int r, int m,int n, int rank);
void RowCorrector2(float *row, int *index, int c);
void RowCorrector(float (*Mx)[c], float (*Mcr), int cRow);
void print(float (*Mp)[c], int r);


int main(int argc, char **argv){

    int rank, size;
    MPI_Init(&argc, &argv);
    int r;
    int flag = 0; //used for checking if there is any process that is sending information to this rank
    int source_rank; //used to check which rank is sending the buffer
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float Mshare[c];
    int MshareI[c];
    MPI_Status status;
    //--------------------------------------------------------------------------------------------------------------------------------------//

    
    if(rank==size-1){
        r = R/size + R%size;
    }
    else{
        r = R/size;
    }
    
    float M1[r][c]; //->is the matrix at the previous time step
    float M2[r][c]; //->is the matrix at the current time step

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
    int time = 0;
    //Infect(M1,r, random_infected,random_infected,rank);
    Infect(M2,r, 4,0,rank); //For Debuging

    if(size>1){
        //####################################################Starting the send block###################################################//
        if(rank==0){
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,1,112,MPI_COMM_WORLD);                                                   //Rank 0 sends the last row to the rank 1
        }
        else if(rank==size-1){                                          
            MPI_Send(&M2[0][0], c,MPI_FLOAT,size-2,112,MPI_COMM_WORLD);                                                //Last rank sends the first row to the second last rank
        }
        else {
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,rank+1,112,MPI_COMM_WORLD);                                              //Inbetween ranks sends first row to the previous rank 
            MPI_Send(&M2[0][0], c,MPI_FLOAT,rank-1,112,MPI_COMM_WORLD);                                                //and the last row to the next rank
        }
        MPI_Barrier(MPI_COMM_WORLD);                                                                                   //Barrier to ensure all ranks have finished sending the rows.

        //##################################################Starting the receive block#################################################//
        if(rank==0){
            MPI_Recv(Mshare, c, MPI_FLOAT, 1, 112, MPI_COMM_WORLD, &status);                                          //Rank 0 receives the first row from the rank 1
            RowCorrector(M2, Mshare, r-1);
        }
        else if(rank==size-1){                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, size-2, 112, MPI_COMM_WORLD, &status);                                     //Last rank receives the last row from the second last rank
            RowCorrector(M2, Mshare, 0);
        }                                                                                                               
        else {                                                                                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, rank+1, 112, MPI_COMM_WORLD, &status);                                    //Inbetween ranks receives last row from the previous rank
            RowCorrector(M2, Mshare, r-1); 
            MPI_Recv(Mshare, c, MPI_FLOAT, rank-1, 112, MPI_COMM_WORLD, &status);                                    //and the first row from the next rank
            RowCorrector(M2, Mshare, 0);
        } 
    }      
//-----------------------------------------For debuging purpose--------------------------------------------------------------//
    if (rank == 0){std::cout<< "Printing the initial stage of the matrix" <<std::endl;}
    for(int node=0;node<size;node++){
        if (node == rank){
        std::cout << "The rank printing is " << rank << std::endl;
        print(M2,r);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
//--------------------------------------------------------------------------------------------------------------------------------------//
    MPI_Barrier(MPI_COMM_WORLD);
    while(time<2){
    //##################################################Reset Recover and Reimmune#################################################//
        ResetRecoverImmune(M2, r, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Iprobe(MPI_ANY_SOURCE,112,MPI_COMM_WORLD, &flag, &status);
        while(flag){
            source_rank = status.MPI_SOURCE;
            MPI_Recv(MshareI, c, MPI_INT, source_rank, 112, MPI_COMM_WORLD, &status);
            //std::cout<<"The sending source is "<<source_rank<<std::endl;
            if(source_rank == rank+1){
                RowCorrector2( M2[r-1], MshareI, c );  //if the sender node is 1+current_node, last row will be modified 
            }
            else if(source_rank == rank-1){
                RowCorrector2( M2[0], MshareI, c );  //if the sender node is 1-current_node, last row will be modified 
            }
            flag = 0;
            MPI_Iprobe(MPI_ANY_SOURCE,112,MPI_COMM_WORLD, &flag, &status); 
        }
        std::cout<<"Rank" << rank<< " COmpleted the Probabing"<<std::endl;

        //##################################################Reinfection Block#################################################//
        for (int i = 0; i < r; i++){         
            for(int j = 0; j < c; j++){
                if(M2[i][j]<1.0){
                    for(int per = 0; per < (int)(M2[i][j]*10); per++){ //per is the person surrounding the cell under consideration
                        resilience = infection_Probability_Distribution(gen);
                        if(0.2>resilience){
                            Infect(M2,r, 3,3,rank); ////CHANGES TO BE MADE HERE ONLY TESTING NOW
                            break;
                        }}}}}
        //--------------------------------------------------------------------------------------------------------------------------------------//

        if(size>1){
            //####################################################Starting the send block###################################################//
            if(rank==0){
                MPI_Send(&M2[r-1][0], c,MPI_FLOAT,1,112,MPI_COMM_WORLD);                                                   //Rank 0 sends the last row to the rank 1
            }
            else if(rank==size-1){                                          
                MPI_Send(&M2[0][0], c,MPI_FLOAT,size-2,112,MPI_COMM_WORLD);                                                //Last rank sends the first row to the second last rank
            }
            else {
                MPI_Send(&M2[r-1][0], c,MPI_FLOAT,rank+1,112,MPI_COMM_WORLD);                                              //Inbetween ranks sends first row to the previous rank 
                MPI_Send(&M2[0][0], c,MPI_FLOAT,rank-1,112,MPI_COMM_WORLD);                                                //and the last row to the next rank
            }
            MPI_Barrier(MPI_COMM_WORLD);                                                                                   //Barrier to ensure all ranks have finished sending the rows.

            //##################################################Starting the receive block#################################################//
            if(rank==0){
                MPI_Recv(Mshare, c, MPI_FLOAT, 1, 112, MPI_COMM_WORLD, &status);                                          //Rank 0 receives the first row from the rank 1
                RowCorrector(M2, Mshare, r-1);
            }
            else if(rank==size-1){                                          
                MPI_Recv(Mshare, c, MPI_FLOAT, size-2, 112, MPI_COMM_WORLD, &status);                                     //Last rank receives the last row from the second last rank
                RowCorrector(M2, Mshare, 0);
            }                                                                                                               
            else {                                                                                                          
                MPI_Recv(Mshare, c, MPI_FLOAT, rank+1, 112, MPI_COMM_WORLD, &status);                                    //Inbetween ranks receives last row from the previous rank
                RowCorrector(M2, Mshare, r-1); 
                MPI_Recv(Mshare, c, MPI_FLOAT, rank-1, 112, MPI_COMM_WORLD, &status);                                    //and the first row from the next rank
                RowCorrector(M2, Mshare, 0);
            } 
        }
        if (rank == 0){std::cout<<"Printing the reinfected stage of the matrix at time "<< time <<std::endl;}
        for(int node=0;node<size;node++){
            if (node == rank){
            std::cout << "The rank printing is " << rank << std::endl;
            print(M2,r);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        //####################################################Adding the current state to history###################################################//
        for (int i = 0; i < r; i++){         
            for(int j= 0; j < c; j++){
                M1[i][j] = M2[i][j];}}
        //-----------------------------------------------------------------------------------------------------------------------------------------//
        time +=1;
        //--------------------------------------------------------------------------------------------------------------------------------------//
        }
        if (rank == 0){std::cout<<"Printing the recovered stage of the matrix"<<std::endl;}
        for(int node=0;node<size;node++){
            if (node == rank){
            std::cout << "The rank printing is " << rank << std::endl;
            print(M2,r);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } 
    
    MPI_Finalize();
    std::cout<<"Execution Complete";
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

void ResetRecoverImmune(float (*Mx)[c], int r, int rank, int size){
    int lastRowRecover[c];
    int firstRowRecover[c];
    std::fill(lastRowRecover, lastRowRecover + c, 0);
    std::fill(firstRowRecover, firstRowRecover + c, 0);
    int trigger[] = {0,0,0,0}; //Indicates rank and the row which was triggered 
    int lsize=0;
    int fsize=0;
	float resilience;
	float recovery_probability;
	int flag;
	std::uniform_real_distribution<float> infection_Probability_Distribution(0.0f, 1.0f);
	for (int i=0;i<r;i++){
        for (int j=0;j<c;j++){
            std::random_device rd;
            std::mt19937 gen(rd());
            recovery_probability = 0.8;//infection_Probability_Distribution(gen);
            if(recovery_probability > q){
                flag = 1;
            }
            else{
                flag = 0;
            }
            if (Mx[i][j] >= 1 && Mx[i][j] < immune_time+1 && flag == 1){
                //#pragma omp parallel for private(i,j) shared (Mx,i,j)
                if (Mx[i][j] == 1){
                    if(rank == 0 && i==r-1){
                        lastRowRecover[lsize++] = j; //Adding the column which was recovered
                        trigger[0] = 1; //Indicating that the rank 0 last row has some changes.
                    }
                    else if(rank == size-1 && i== 0){
                        firstRowRecover[fsize++] = j; //Adding the column which was recovered
                        trigger[1] = 1; //Indicating that the rank 1 first row has some changes.
                    }
                    else if(rank != size-1 && rank != 0){
                        if(i == 0){
                            firstRowRecover[fsize++] = j; //Adding the column which was recovered
                            trigger[2] = 1; //Indicating that the random rank first row has some changes.
                        }
                        if(i == r-1){
                            lastRowRecover[lsize++] = j; //Adding the column which was recovered
                            trigger[3] = 1; //Indicating that the random rank last row has some changes.
                        }
                    }
                }
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
    if(trigger[0]==1){
        MPI_Send(lastRowRecover,c,MPI_INT,1,112,MPI_COMM_WORLD);
    }
    else if(trigger[1]==1){
        MPI_Send(firstRowRecover,c,MPI_INT,size-2,112,MPI_COMM_WORLD);
    }
    else if(trigger[2]==1){
        MPI_Send(firstRowRecover,c,MPI_INT,rank-1,112,MPI_COMM_WORLD);
    }
    else if(trigger[3]==1){
        MPI_Send(lastRowRecover,c,MPI_INT,rank+1,112,MPI_COMM_WORLD);
    }
}

void RowCorrector2(float *row, int *index, int c){
    //index is the array where the numbers of the Infected cells from the previous rank is stored.
    for(int i=0;i<c;i++){
        if (i>0 && index[i] == 0){
            break;
        }
        if(index[i] == 0){
            if(row[index[i]] < 1.0 && row[index[i]] > 0.0){
                row[index[i]] = row[index[i]] - 0.1;
            }
            if(row[index[i]+1] < 1.0 && row[index[i]+1] > 0.0){
                row[index[i]+1] = row[index[i]] - 0.1;
            }
        }
        else if(index[i] == c-1){
            if(row[index[i]] < 1.0 && row[index[i]] > 0.0){
                row[index[i]] = row[index[i]] - 0.1;
            }
            if(row[index[i]-1] < 1.0 && row[index[i]-1] > 0.0){
                row[index[i]-1] = row[index[i]] - 0.1;
            }
        }
        else {
            if(row[index[i]-1] < 1.0 && row[index[i]-1] > 0.0){
                row[index[i]-1] = row[index[i]] - 0.1;
            }
            if(row[index[i]] < 1.0 && row[index[i]] > 0.0){
                row[index[i]] = row[index[i]] - 0.1;
            }
            if(row[index[i]+1] < 1.0 && row[index[i]+1] > 0.0){
                row[index[i]+1] = row[index[i]+1] - 0.1;
            }
        }
    }
}

void Infect(float (*Mx)[c], int r, int m,int n, int rank){
    int tr,tc,er,ec;
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
