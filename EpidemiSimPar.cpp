#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include "mpi.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>  
#include <ctime>    

/*
======================================================= ****Universal Constants**** =======================================================
-------------------------------------------------------------------------------------------------------------------------------------------
The below defined are some of the universal constants used in this Program which are common to all the Ranks.
-------------------------------------------------------------------------------------------------------------------------------------------
    q                       = Threshold for recovery of the infected cell.
    probabilityOfInfection  = Threshold for a cell to get infected due to neighbour.
    immune_time             = Time steps untill which the cell is immune to the infection after it was once recovered
    R                       = Total number of rows of the grid under consideration
    c                       = Total number of columns of the grid under consideration
___________________________________________________________________________________________________________________________________________    
*/
const static float q                        = 0.1;
const static float probabilityOfInfection   = 0.5;
const static int immune_time                = 3;
const static int R                          = 25; 
const static int c                          = 25;                                                                                     

/*
======================================================= ****Function Declaration**** ======================================================
-------------------------------------------------------------------------------------------------------------------------------------------
The below defined are some of the universal fuction used in this Program which are common to all the Ranks.
For specific details about a specific function please see the description in the function itself.
-------------------------------------------------------------------------------------------------------------------------------------------
    printMatrixToFile  = Prints the matrix of the specific rank in a specific text file with the name matrix_output_rank_<rank_number>.txt.
            Required inputs: - The memory address of the 2D grid, number of rank specific rows, total number of columns and filename.      
    ---------------------------------------------------------------------------------------------------------------------------------------
    ResetRecoverImmune = Runs the recovery of the infected cells and sets the immunity on the recovered cells.
            Required inputs: - The memory address of the 2D grid, number of rank specific rows, the rank number, total number of ranks.
    ---------------------------------------------------------------------------------------------------------------------------------------
    Infect             = Sets the infection on a cell, if the probability of infection is greater than given probabilityOfInfection.
            Required inputs: - The memory address of the 2D grid, number of rank specific rows, row number of infected cell, 
                               column number of infected cell, the rank number.
    ---------------------------------------------------------------------------------------------------------------------------------------
    RowCorrector2      = Corrects the first and the last row of the current matrix after recovery depending on the current rank number,  
                         based on last and first row of previous and next rank matrix respectively.
            Required inputs: - The memory address of first/last row of the 2D grid, array with the information on recovery location, 
                               total number of columns, the rank number.
    ---------------------------------------------------------------------------------------------------------------------------------------
    RowCorrector       = Corrects the first and the last row of the current matrix after infection depending on the current rank number,  
                         based on last and first row of previous and next rank matrix respectively.
            Required inputs: - The memory address of the 2D grid, array with the information on first/last row of next/previous rank grid, 
                               total number of columns.
___________________________________________________________________________________________________________________________________________    
*/
void printMatrixToFile(float (*matrix)[c], int rows, int cols, const std::string& filename);
void ResetRecoverImmune(float (*Mx)[c], int r, int rank, int size,int timet);
void Infect(float (*Mx)[c], int r, int m,int n, int rank);
void RowCorrector2(float *row, int *index, int c, int rank);
void RowCorrector3(float *row, int *index, int c, int rank);
void RowCorrector(float (*Mx)[c], float (*Mcr), int cRow);

//____________________________________________________ **** End of Declarations **** _____________________________________________________//
//_______________________________________________________________________________________________________________________________________//

int main(int argc, char **argv){
/*
======================================================== ****The Main Function**** ========================================================
-------------------------------------------------------------------------------------------------------------------------------------------
The start point of our simulation code.
-------------------------------------------------------------------------------------------------------------------------------------------
  Variables Used: -
  -----------------
    rank                -> Current rank number
    size                -> Total number of ranks
    r                   -> Number of rows alloted to this rank from the total rows of the rows in the Global Matrix
    M1                  -> State of the cells in the previous time step
    M2                  -> State of the cells in the current time step
    Mshare              -> Stores the row sent from another rank after infection, used in the MPI_Recv function call
    MshareI             -> Stores the row sent from another rank after recovery, used in the MPI_Recv function call
    MshareF             -> Stores the information about infection location, used in the MPI_Recv function call
    random_infected     -> Random row number for initial infection
    random_infectedc    -> Random column number for initial infection
    resilience          -> Resilience against infection
    time                -> Current time step. 

___________________________________________________________________________________________________________________________________________    
*/
    int rank, size,r,i,j,nthreads;
    MPI_Init(&argc, &argv);
    double starttime, endtime;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    starttime = MPI_Wtime();
    if(rank==0){
        std::cout<<"The Simulation started with total ranks = "<< size <<std::endl;
    }
    float Mshare[c+1];
    int MshareI[c+1];
    int MshareF[c+1];
    MPI_Status status;
    std::string filename = "matrix_output_rank_" + std::to_string(rank) + ".txt";
//------------------------------------------------------------------------------------------------------------------------------------------//

    if(rank==size-1){
        r = R/size + R%size;
    }
    else{
        r = R/size;
    } 
    float M1[r][c]; 
    float M2[r][c]; 

//--------------------------------------------------Initilizing the matrices---------------------------------------------------------------//
    #pragma omp parallel private(i)
    {
        nthreads = omp_get_num_threads();
        if(rank==0){
            #pragma omp master
            {
                std::cout<<"The simulation started with a total number of threads = "<< nthreads<<std::endl;
            }
        }
        #pragma omp for private(i,j)
            for(i=0;i<r;i++){
                for(j=0;j<c;j++){
                    M1[i][j]=0;
                    M2[i][j]=0;
                }
            }
    }
//----------------------------------------------------------------------------------------------------------------------------------------//

    auto time_now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    unsigned int seed = static_cast<unsigned int>(time_now*rank);
    std::mt19937 gen(seed);                                                               // Initialize the Mersenne Twister random number generator
    std::uniform_int_distribution<int> distribution(0, r-1);                                 
    std::uniform_real_distribution<float> infection_Probability_Distribution(0.0f, 1.0f);
    std::uniform_int_distribution<int> distribution2(0, c-1);                               
                                                                                                                        
    int random_infected = distribution(gen);
    int random_infectedc = distribution2(gen);                                            // Generate a random number
    float resilience;
    int lsize = 0;
    int fsize = 0;
    int per = 0;
//------------------------------------------------------------------Initial Infection-----------------------------------------------------//
    int timet = 0;
    int mythread;
    #pragma omp parallel for private(mythread,i,random_infected,random_infectedc,gen) shared (M2)
        for(i=0;i<5;i++){
            random_infected = distribution(gen);
            random_infectedc = distribution2(gen);
            if(M2[random_infected][random_infectedc]<1){
                Infect(M2,r, random_infected, random_infectedc,rank);}}
    MPI_Barrier(MPI_COMM_WORLD);
//--------------------------------------------------------------------End of Initial Infection------------------------------------------//
    //printMatrixToFile(M2, r, c, filename);
    if(size>1){
//####################################################################Starting the send block###########################################//
        if(rank==0){
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,1,112,MPI_COMM_WORLD);       //Rank 0 sends the last row to the rank 1
        }
        else if(rank==size-1){                                          
            MPI_Send(&M2[0][0], c,MPI_FLOAT,size-2,112,MPI_COMM_WORLD);    //Last rank sends the first row to the second last rank
        }
        else {
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,rank+1,112,MPI_COMM_WORLD);  //Inbetween ranks sends first row to the previous rank 
            MPI_Send(&M2[0][0], c,MPI_FLOAT,rank-1,112,MPI_COMM_WORLD);    //and the last row to the next rank
        }
        MPI_Barrier(MPI_COMM_WORLD);                                       //Barrier to ensure all ranks have finished sending the rows.

//####################################################################Starting the receive block########################################//
        if(rank==0){
            MPI_Recv(Mshare, c, MPI_FLOAT, 1, 112, MPI_COMM_WORLD, &status);      //Rank 0 receives the first row from the rank 1
            RowCorrector(M2, Mshare, r-1);
        }
        else if(rank==size-1){                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, size-2, 112, MPI_COMM_WORLD, &status); //Last rank receives the last row from the second last rank
            RowCorrector(M2, Mshare, 0);
        }                                                                                                               
        else {                                                                                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, rank+1, 112, MPI_COMM_WORLD, &status); //Inbetween ranks receives last row from the previous rank
            RowCorrector(M2, Mshare, r-1); 
            MPI_Recv(Mshare, c, MPI_FLOAT, rank-1, 112, MPI_COMM_WORLD, &status); //and the first row from the next rank
            RowCorrector(M2, Mshare, 0);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #pragma omp parallel for private(i,j) shared (M1,M2,r,c)
        for (i = 0; i < r; i++){         
            for(j= 0; j < c; j++){
                M1[i][j] = M2[i][j];}}                                            //Copying the infected state
//-------------------------------------------------------------------------For debuging purpose------------------------------------------//
    if (rank == 0){std::cout<< "Completed the initial stage of the matrix" <<std::endl;}
    printMatrixToFile(M2, r, c, filename);
    MPI_Barrier(MPI_COMM_WORLD);
//--------------------------------------------------------------------------------------------------------------------------------------//    
    while(timet<4){
//##########################################################################Reset Recover and Reimmune##################################//
        if(timet != 0){
            ResetRecoverImmune(M2, r, rank, size, timet);
            MPI_Barrier(MPI_COMM_WORLD);
            if(size>1){
                if(rank == 0){
                    MPI_Recv(MshareI, c+1, MPI_INT,1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] != 0){
                        RowCorrector2(M2[r-1], MshareI, c,rank);
                    }
                }
                else if(rank == size-1){
                    MPI_Recv(MshareI, c+1, MPI_INT,size-2,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] != 0){
                        RowCorrector2(M2[0], MshareI, c, rank);
                    }
                }
                else{
                    MPI_Recv(MshareI, c+1, MPI_INT,rank-1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] == 1){
                        RowCorrector2(M2[r-1], MshareI, c, rank);
                    }
                    else if(MshareI[c] == 2){
                        RowCorrector2(M2[0], MshareI, c, rank);
                    }
                    MPI_Recv(MshareI, c+1, MPI_INT,rank+1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] == 1){
                        RowCorrector2(M2[r-1], MshareI, c, rank);
                    }
                    else if(MshareI[c] == 2){
                        RowCorrector2(M2[0], MshareI, c, rank);
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } 
        std::fill(MshareF, MshareF + c + 1, 0);   //Mshare for first row
        std::fill(MshareI, MshareI + c + 1, 0); //MshareI for last row
        lsize = 0;
        fsize = 0;
//############################################################################Reinfection Block#########################################//        
        for (i = 0; i < r; i++){         
            for(j = 0; j < c; j++){
                if(M1[i][j]<1){  //Passauf. here we used M1 as our reference as infection happens based on the previous time step
                    for(per = 0; per < (int)(M1[i][j]*10); per++){ //per is the person surrounding the cell under consideration
                        resilience = infection_Probability_Distribution(gen);
                        if(probabilityOfInfection > resilience){
                            #pragma omp critical
                            {    
                                Infect(M2,r, i,j,rank);
                                if(i==0){
                                    MshareF[fsize++] = j;
                                    MshareF[c] = r-1;
                                }
                                else if(i==r-1){
                                    MshareI[lsize++] = j;
                                    MshareI[c] = r-1;
                                }
                            }
                            break;
                            }}}}}
        MPI_Barrier(MPI_COMM_WORLD);
//----------------------------------------------------------------------------------------------------------------------------------------//
        if(size>1){
            //####################################################Starting the send block###################################################//
            if(rank==0){
                MPI_Send(&MshareI, c+1, MPI_INT, 1, 112, MPI_COMM_WORLD);                                                   //Rank 0 sends the last row to the rank 1
            }
            else if(rank==size-1){                                          
                MPI_Send(&MshareF, c+1, MPI_INT, size-2, 112, MPI_COMM_WORLD);                                                //Last rank sends the first row to the second last rank
            }
            else {
                MPI_Send(&MshareI, c+1, MPI_INT, rank+1, 112, MPI_COMM_WORLD);                                              //Inbetween ranks sends first row to the previous rank 
                MPI_Send(&MshareF, c+1, MPI_INT, rank-1, 112, MPI_COMM_WORLD);                                                //and the last row to the next rank
            }
            MPI_Barrier(MPI_COMM_WORLD);                                                                                   //Barrier to ensure all ranks have finished sending the rows.

            //##################################################Starting the receive block#################################################//
            if(rank==0){
                MPI_Recv(MshareI, c+1, MPI_INT, 1, 112, MPI_COMM_WORLD, &status);                                          //Rank 0 receives the first row from the rank 1
                if(MshareI[c]>0){
                    RowCorrector3(M2[r-1], MshareI, c, rank);
                }
            }   
            else if(rank==size-1){                                         
                MPI_Recv(MshareF, c+1, MPI_INT, size-2, 112, MPI_COMM_WORLD, &status);                                     //Last rank receives the last row from the second last rank
                if(MshareF[c]>0){
                    RowCorrector3(M2[0], MshareF, c, rank);
                }
            }                                                                                                               
            else {                                                                                                          
                MPI_Recv(MshareI, c+1, MPI_INT, rank+1, 112, MPI_COMM_WORLD, &status);                                    //Inbetween ranks receives last row from the previous rank
                if(MshareI[c]>0){
                    RowCorrector3(M2[r-1], MshareI, c, rank);
                } 
                MPI_Recv(MshareF, c+1, MPI_INT, rank-1, 112, MPI_COMM_WORLD, &status);                                    //and the first row from the next rank
                if(MshareF[c]>0){
                    RowCorrector3(M2[0], MshareF, c, rank);
                }
            } 
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //####################################################Adding the current state to history###################################################//
        #pragma omp parallel for private(i,j) shared (M1,M2,r,c)
            for (i = 0; i < r; i++){         
                for(j= 0; j < c; j++){
                    M1[i][j] = M2[i][j];}}
        //-----------------------------------------------------------------------------------------------------------------------------------------//
        timet +=1; //updating the time
        if (rank == 0){std::cout<< "Completed the stage of the matrix at time t = "<< timet <<std::endl;}
	    printMatrixToFile(M2,r,c,filename);
        //--------------------------------------------------------------------------------------------------------------------------------------//
        }
        if(rank==0){
        std::cout<<"The Simulation completed with no errors."<<std::endl;
        std::cout<<"All the necessary files have been saved successfully."<<std::endl;
            }
        endtime = MPI_Wtime();
        std::cout<<"The time taken is "<<endtime-starttime<<" for "<<size<<" number of ranks"<< std::endl;
    MPI_Finalize();
    return 0;
}

//________________________________________________________________________________________________________________________________________________//

void RowCorrector(float (*Mx)[c], float (*Mcr), int cRow){
    int i=0;
        for(i=0;i<c;i++){
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

void ResetRecoverImmune(float (*Mx)[c], int r, int rank, int size,int timet){ 
    int lastRowRecover[c+1];
    int firstRowRecover[c+1];
    int k,l;
    std::fill(lastRowRecover, lastRowRecover + c + 1, 0);
    std::fill(firstRowRecover, firstRowRecover + c + 1, 0); 
    int lsize = 0;
    int fsize = 0;
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
                if (Mx[i][j] == 1){
                    if(rank == 0 && i==r-1){
                        lastRowRecover[lsize++] = j; //Adding the column which was recovered
                        lastRowRecover[c] = 2;
                    }
                    else if(rank == size-1 && i== 0){
                        firstRowRecover[fsize++] = j; //Adding the column which was recovered
                        firstRowRecover[c] = 1; //Indicating that the rank 1 first row has some changes.
                    }
                    else if(rank != size-1 && rank != 0){
                        if(i == 0){
                            firstRowRecover[fsize++] = j; //Adding the column which was recovered
                            firstRowRecover[c] = 1; //Indicating that the random rank first row has some changes.
                        }
                        if(i == r-1){
                            lastRowRecover[lsize++] = j; //Adding the column which was recovered
                            lastRowRecover[c] = 2; //Indicating that the random rank last row has some changes.
                        }
                    }
                }
                #pragma omp parallel for private(k,l) shared (Mx,i,j)
                for (k=i-1;k<i+2;k++){
                    for (l=j-1;l<j+2;l++){
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
        }
    }
    if(size > 1){
        if(rank == 0){
            MPI_Send(lastRowRecover,c+1,MPI_INT,1,112,MPI_COMM_WORLD);
        }
        else if(rank == size-1){
            MPI_Send(firstRowRecover,c+1,MPI_INT,size-2,112,MPI_COMM_WORLD);
        }
        else {
            MPI_Send(firstRowRecover,c+1,MPI_INT,rank-1,112,MPI_COMM_WORLD);
            MPI_Send(lastRowRecover,c+1,MPI_INT,rank+1,112,MPI_COMM_WORLD);
        }
    }
}

void RowCorrector2(float *row, int *index, int c, int rank){
    int i=0;
        for(i=0;i<c;i++){
            if (i>0 && index[i] == 0){
                break;
            }
            if(index[i] == 0){
                if(row[index[i]] < 1.0 && row[index[i]] > 0.0){
                    row[index[i]] = row[index[i]] - 0.1;
                }
                if(row[index[i]+1] < 1.0 && row[index[i]+1] > 0.0){
                    row[index[i]+1] = row[index[i]+1] - 0.1;
                }
            }
            else if(index[i] == c-1){
                if(row[index[i]] < 1.0 && row[index[i]] > 0.0){
                    row[index[i]] = row[index[i]] - 0.1;
                }
                if(row[index[i]-1] < 1.0 && row[index[i]-1] > 0.0){
                    row[index[i]-1] = row[index[i]-1] - 0.1;
                }
            }
            else {
                if(row[index[i]-1] < 1.0 && row[index[i]-1] > 0.0){
                    row[index[i]-1] = row[index[i]-1] - 0.1;
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

void RowCorrector3(float *row, int *index, int c, int rank){
    int i=0;
        for(i=0;i<c;i++){
            if (i>0 && index[i] == 0){
                break;
            }
            if(index[i] == 0){
                if(row[index[i]] < 1.0){
                    row[index[i]] = row[index[i]] + 0.1;
                }
                if(row[index[i]+1] < 1.0){
                    row[index[i]+1] = row[index[i]+1] + 0.1;
                }
            }
            else if(index[i] == c-1){
                if(row[index[i]] < 1.0){
                    row[index[i]] = row[index[i]] + 0.1;
                }
                if(row[index[i]-1] < 1.0){
                    row[index[i]-1] = row[index[i]-1] + 0.1;
                }
            }
            else {
                if(row[index[i]-1] < 1.0){
                    row[index[i]-1] = row[index[i]-1] + 0.1;
                }
                if(row[index[i]] < 1.0){
                    row[index[i]] = row[index[i]] + 0.1;
                }
                if(row[index[i]+1] < 1.0){
                    row[index[i]+1] = row[index[i]+1] + 0.1;
                }
            }
        }
    
}

void Infect(float (*Mx)[c], int r, int m,int n, int rank){
    int tr,tc,er,ec,i,j;
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

    #pragma omp parallel for private(i,j) shared (tr,tc,er,ec,Mx)
    for(i=tr;i<er;i++){
        for(j=tc;j<ec;j++){
            if (Mx[i][j] < 1 ){
                Mx[i][j] = Mx[i][j] + 0.1;
            } 
        }
    }
    if(Mx[m][n]<=1){
        Mx[m][n] = 1;
    }
}

// Function to print a matrix to a text file
void printMatrixToFile(float (*matrix)[c], int rows, int cols, const std::string& filename) {
    
    std::ofstream outFile(filename, std::ios::app);
    if (!outFile) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if(matrix[i][j]<0.1){
            outFile << std::setw(4) << 0 << ";";
            }
            else {
                    outFile << std::setw(4) << matrix[i][j] << ";";  
            }
        }
        outFile << std::endl;
    }
    outFile << std::endl;  
    outFile.close();
}