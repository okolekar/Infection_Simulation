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
const static int R                          = 2000; 
const static int c                          = 2000;                                                                                     

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
void printRowToFile(float* matrix, int cols, int timet, const std::string& filename);
void printRowToFileInt(int (*matrix), int cols, int timet, const std::string& filename);
unsigned int custom_lcg(unsigned int seed) {
    return (1103515245 * seed + 12345) % (1 << 31);
}

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
    int required=1, provided;
    MPI_Init_thread(&argc,&argv,required,&provided);
    double starttime, endtime;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    starttime = MPI_Wtime();
    if(rank==0){
        std::cout<<"The Simulation started with total ranks = "<< size <<std::endl;
        std::cout<<"Level provided: "<<provided<<" and the required level is "<<required<<std::endl; 
    }
    float Mshare[c+1];
    int MshareI[c+1];
    int MshareF[c+1];
    MPI_Status status;
    std::string filename = "matrix_output_rank_" + std::to_string(rank) + ".txt";
    std::string sentrowfilename = "sent_output_rank_" + std::to_string(rank) + ".txt";
    std::string recivfilename = "recieve_output_rank_" + std::to_string(rank) + ".txt";
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
    std::random_device rd;
    unsigned int seed = static_cast<unsigned int>(time_now) ^ (rank + 1) * 101 ^ rd();
    std::mt19937 gen(seed);                                                               // Initialize the Mersenne Twister random number generator
    std::uniform_real_distribution<float> infection_Probability_Distribution(0.0f, 1.0f);                                 
    unsigned int random_seed = custom_lcg(seed);                                                                                                                                                  
    int random_infected = random_seed % r;
    random_seed = custom_lcg(random_seed);
    int random_infectedc = random_seed % c;                                            // Generate a random number
    float resilience;
    int lsize = 0;
    int fsize = 0;
    int per = 0;
//------------------------------------------------------------------Initial Infection-----------------------------------------------------//
    int timet = 0;
    int mythread;
    #pragma omp parallel for private(mythread,i,random_infected,random_infectedc,gen) shared (M2)
        for(i=0;i<5;i++){
            random_seed = custom_lcg(seed+i);
            random_infected = random_seed % (r-1);
            if(random_infected>r-1){random_infected = r-1;}
            random_seed = custom_lcg(seed+i);
            random_infectedc = random_seed % (c-1);
            if(random_infectedc>c-1){random_infectedc = c-1;}
            if(M2[random_infected][random_infectedc]<1){
                //std::cout<<"The seed is "<< seed << " the random_infected and random_infectedc is "<< random_infected << " " << random_infectedc << " and rank is "<< rank << std::endl;
                Infect(M2,r, random_infected, random_infectedc,rank);}}
    MPI_Barrier(MPI_COMM_WORLD);
//--------------------------------------------------------------------End of Initial Infection------------------------------------------//
    //printMatrixToFile(M2, r, c, filename);
    if(size>1){
//####################################################################Starting the send block###########################################//
        if(rank==0){
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,1,112,MPI_COMM_WORLD);       //Rank 0 sends the last row to the rank 1
            //printRowToFile(&M2[r-1][0], c, timet, sentrowfilename);
        }
        else if(rank==size-1){                                          
            MPI_Send(&M2[0][0], c,MPI_FLOAT,size-2,112,MPI_COMM_WORLD);    //Last rank sends the first row to the second last rank
            //printRowToFile(&M2[0][0], c+1, timet, sentrowfilename);
        }
        else {
            MPI_Send(&M2[r-1][0], c,MPI_FLOAT,rank+1,112,MPI_COMM_WORLD);  //Inbetween ranks sends first row to the previous rank 
            //printRowToFile(&M2[r-1][0], c+1, timet, sentrowfilename);
            MPI_Send(&M2[0][0], c,MPI_FLOAT,rank-1,112,MPI_COMM_WORLD);    //and the last row to the next rank
            //printRowToFile(&M2[0][0], c+1, timet, sentrowfilename);
        }
        MPI_Barrier(MPI_COMM_WORLD);                                       //Barrier to ensure all ranks have finished sending the rows.

//####################################################################Starting the receive block########################################//
        if(rank==0){
            MPI_Recv(Mshare, c, MPI_FLOAT, 1, 112, MPI_COMM_WORLD, &status);      //Rank 0 receives the first row from the rank 1
            RowCorrector(M2, Mshare, r-1);
            //printRowToFile(Mshare, c, timet, recivfilename);
        }
        else if(rank==size-1){                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, size-2, 112, MPI_COMM_WORLD, &status); //Last rank receives the last row from the second last rank
            RowCorrector(M2, Mshare, 0);
            //printRowToFile(Mshare, c, timet, recivfilename);
        }                                                                                                               
        else {                                                                                                          
            MPI_Recv(Mshare, c, MPI_FLOAT, rank+1, 112, MPI_COMM_WORLD, &status); //Inbetween ranks receives last row from the previous rank
            RowCorrector(M2, Mshare, r-1);
            //printRowToFile(Mshare, c, timet, recivfilename); 
            MPI_Recv(Mshare, c, MPI_FLOAT, rank-1, 112, MPI_COMM_WORLD, &status); //and the first row from the next rank
            RowCorrector(M2, Mshare, 0);
            //printRowToFile(Mshare, c, timet, recivfilename);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #pragma omp parallel for private(i,j) shared (M1,M2,r,c)
        for (i = 0; i < r; i++){         
            for(j= 0; j < c; j++){
                M1[i][j] = M2[i][j];}}                                            //Copying the infected state
//-------------------------------------------------------------------------For debuging purpose------------------------------------------//
    if (rank == 0){std::cout<< "Completed the initial stage of the matrix" <<std::endl;}
    //printMatrixToFile(M2, r, c, filename);
    MPI_Barrier(MPI_COMM_WORLD);
//--------------------------------------------------------------------------------------------------------------------------------------//    
    while(timet<50){
//##########################################################################Reset Recover and Reimmune##################################//
        if(timet != 0){
            ResetRecoverImmune(M2, r, rank, size, timet);
            MPI_Barrier(MPI_COMM_WORLD);
            if(size>1){
                if(rank == 0){
                    MPI_Recv(MshareI, c+1, MPI_INT,1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] != 0){
                        RowCorrector2(M2[r-1], MshareI, c,rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                    }
                }
                else if(rank == size-1){
                    MPI_Recv(MshareI, c+1, MPI_INT,size-2,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] != 0){
                        RowCorrector2(M2[0], MshareI, c, rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                    }
                }
                else{
                    MPI_Recv(MshareI, c+1, MPI_INT,rank-1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] == 1){
                        RowCorrector2(M2[r-1], MshareI, c, rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                    }
                    else if(MshareI[c] == 2){
                        RowCorrector2(M2[0], MshareI, c, rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                    }
                    MPI_Recv(MshareI, c+1, MPI_INT,rank+1,112,MPI_COMM_WORLD,&status);
                    if(MshareI[c] == 1){
                        RowCorrector2(M2[r-1], MshareI, c, rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                    }
                    else if(MshareI[c] == 2){
                        RowCorrector2(M2[0], MshareI, c, rank);
                        //printRowToFileInt(MshareI, c+1, timet, recivfilename);
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
        #pragma omp parallel for private(per,i,j,gen,resilience) shared (M2,M1,r,c,rank,fsize,lsize,MshareI,MshareF)
        for (i = 0; i < r; i++){         
            for(j = 0; j < c; j++){
                if(M1[i][j]<1){  //Passauf. here we used M1 as our reference as infection happens based on the previous time step
                    for(per = 0; per < (int)(M1[i][j]*10); per++){ //per is the person surrounding the cell under consideration
                        resilience = infection_Probability_Distribution(gen);
                        if(probabilityOfInfection > resilience){
                            #pragma omp critical
                            {    
                                if(i==0){
                                    #pragma omp master
                                    {
                                        Infect(M2,r, i,j,rank);
                                        MshareF[fsize++] = j;
                                        MshareF[c] = r-1;
                                    }
                                }
                                else if(i==r-1){
                                    #pragma omp master
                                    {   
                                        /*std::ofstream outFile(sentrowfilename, std::ios::app);
                                        if (!outFile) {
                                            std::cerr << "Unable to open file " << sentrowfilename << std::endl;
                                        }
                                        outFile << "The time step was " << timet << "and I have added j = "<< j<<" to the MshareI now check the sent row"  << std::endl;
                                        outFile.close();*/
                                        Infect(M2,r, i,j,rank);
                                        MshareI[lsize++] = j;
                                        MshareI[c] = r-1;
                                    }
                                }
                                else {
                                    Infect(M2,r, i,j,rank);
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
                //printRowToFileInt(MshareI, c+1, timet, sentrowfilename);
            }
            else if(rank==size-1){                                          
                MPI_Send(&MshareF, c+1, MPI_INT, size-2, 112, MPI_COMM_WORLD);                                                //Last rank sends the first row to the second last rank
                //printRowToFileInt(MshareF, c+1, timet, sentrowfilename);
            }
            else {
                MPI_Send(&MshareI, c+1, MPI_INT, rank+1, 112, MPI_COMM_WORLD);                                              //Inbetween ranks sends first row to the previous rank 
                //printRowToFileInt(MshareI, c+1, timet, sentrowfilename);
                MPI_Send(&MshareF, c+1, MPI_INT, rank-1, 112, MPI_COMM_WORLD);                                                //and the last row to the next rank
                //printRowToFileInt(MshareF, c+1, timet, sentrowfilename);
            }
            MPI_Barrier(MPI_COMM_WORLD);                                                                                   //Barrier to ensure all ranks have finished sending the rows.

            //##################################################Starting the receive block#################################################//
            if(rank==0){
                MPI_Recv(MshareI, c+1, MPI_INT, 1, 112, MPI_COMM_WORLD, &status);                                          //Rank 0 receives the first row from the rank 1
                if(MshareI[c]>0){
                    RowCorrector3(M2[r-1], MshareI, c, rank);
                    //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                }
            }   
            else if(rank==size-1){                                         
                MPI_Recv(MshareF, c+1, MPI_INT, size-2, 112, MPI_COMM_WORLD, &status);                                     //Last rank receives the last row from the second last rank
                if(MshareF[c]>0){
                    RowCorrector3(M2[0], MshareF, c, rank);
                    //printRowToFileInt(MshareF, c+1, timet, recivfilename);
                }
            }                                                                                                               
            else {                                                                                                          
                MPI_Recv(MshareI, c+1, MPI_INT, rank+1, 112, MPI_COMM_WORLD, &status);                                    //Inbetween ranks receives last row from the previous rank
                if(MshareI[c]>0){
                    RowCorrector3(M2[r-1], MshareI, c, rank);
                    //printRowToFileInt(MshareI, c+1, timet, recivfilename);
                } 
                MPI_Recv(MshareF, c+1, MPI_INT, rank-1, 112, MPI_COMM_WORLD, &status);                                    //and the first row from the next rank
                if(MshareF[c]>0){
                    RowCorrector3(M2[0], MshareF, c, rank);
                    //printRowToFileInt(MshareI, c+1, timet, recivfilename);
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
	    //printMatrixToFile(M2,r,c,filename);
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
    #pragma omp parallel for private(i) shared (Mx,Mcr,cRow,c)
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
void printRowToFile(float* matrix, int cols, int timet, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::app);
    if (!outFile) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }
    outFile << "The time step was " << timet << std::endl;
        for (int j = 0; j < cols; ++j) {
            if(matrix[j]<0.1){
            outFile << std::setw(4) << 0 << ";";
            }
            else {
                    outFile << std::setw(4) << matrix[j] << ";";  
            }
        }
    outFile << std::endl;  
    outFile.close();
}

void printRowToFileInt(int (*matrix), int cols, int timet, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::app);
    if (!outFile) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }
    outFile << "The time step was " << timet << std::endl;
        for (int j = 0; j < cols; ++j) {
            if(matrix[j]<0.1){
            outFile << std::setw(4) << 0 << ";";
            }
            else {
                    outFile << std::setw(4) << matrix[j] << ";";  // Format the output as needed
            }
        }
    outFile << std::endl;  
    outFile.close();
}
