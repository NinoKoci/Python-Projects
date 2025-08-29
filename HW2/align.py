import numpy as np 
import sys

def parse_FASTA_file(fasta_file):
    sequences = [] # the list we will store the sequences 
    temp_list = [] # temporary list for looping each line 
    
    for line in fasta_file: 
        line = line.strip()#remove the /m
        if line.startswith('>'): #initiate loop with searching for the headers 
            if temp_list: #if the list is not empty, append the elements to the sequences list
                sequences.append(''. join(temp_list))
            temp_list=[]   # empty the list after you are done, so that it can loop the next set following a header confrmation 
        else:
            temp_list.append(line) #return the line element to loop it back so it can go back to the original
    if temp_list:  
        sequences.append(''.join(temp_list)) #make sure to include the last element 
    return sequences    
    x = sequences[0]
    y = sequences[1]# #split the list into two, for both x and y 
    return(x,y)

class GlobalSeqAlignment:
    def __init__(self, x, y, M, m, g):#initialize the class with the sequences and the scoring system
        self.M = M
        self.m = m
        self.g = g
        self.x = x
        self.y = y

    def matchmaker(self, x, y):#define the match and mismatch pennalty 
        if x == y:
            return self.M #return Match
        elif x!=y:
            return self.m #return mismatch 

    def matrix_builder(self):
        score_matrix = np.zeros((len(self.x)+1, len(self.y)+1)) #initialize a score matrix with zeros
        traceback_matrix = np.empty(shape =((len(self.x)+1, len(self.y)+1)), dtype = 'object') #initialize traceback matrix with empty strings
        for i in range(1,len(self.x)+1):
            for j in range(1,len(self.y)+1): #start after the first row and column, since we can still fill in the matricies 
                #initialize the first scoring and traceback positions in their matrices 
                score_matrix[0][0] = 0 #initialize the first starting point between two gaps in the sore matrix
                traceback_matrix[0][0] = 'start' #initialize the starting position in the traceback matrix
                score_matrix[0][j] = score_matrix[0][j-1] + self.g #score for gaps in the rows
                traceback_matrix[0][j] = '<-' #add the vertical gap traceback in the first row 
                score_matrix[i][0] = score_matrix[i-1][0] + self.g #score for gaps in the columns 
                traceback_matrix[i][0] = '^' #add the vertical gap traceback in the first column 
                #score the matrix based on 
                match_or_missmatch = score_matrix[i-1][j-1] + GlobalSeqAlignment.matchmaker(self,self.x[i-1], self.y[j-1]) #socre for match or mismatch
                vertical_gap = score_matrix[i-1][j] + self.g
                horizontal_gap = score_matrix[i][j-1] + self.g #score for the gaps
                score_matrix[i][j]=max(match_or_missmatch,vertical_gap,horizontal_gap)
                #do the traceback for the matrix, prioritize the match or mismatch over the gaps. ensure trhe verital gfap is prioritized over the horizontal gap as well
                if score_matrix[i][j] == match_or_missmatch:
                    traceback_matrix[i][j] = '\\'
                elif score_matrix[i][j] == vertical_gap:
                    traceback_matrix[i][j] = '^'
                elif score_matrix[i][j] == horizontal_gap:
                    traceback_matrix[i][j] = '<-'
        return traceback_matrix #we don't need the score matrix 

            
    def get_optimal_alignment(self):
        traceback_matrix = GlobalSeqAlignment.matrix_builder(self) #return the traceback matrix for traceback
        x_prime = []
        y_prime = [] #set up the empty strings for the optimal alignments
        i,j = len(self.x), len(self.y)#define the indexes for the traceback matrix and the sequences
    
        while i and j >0: #input the conditions for the indexes
            #traceback from the bottom right corner of the matrix
            if traceback_matrix[i][j]== '\\': 
                x_prime.append(self.x[i-1])
                y_prime.append(self.y[j-1])
                i -=1
                j -=1
                
            elif traceback_matrix[i][j]== '<-':
                x_prime.append('-')
                y_prime.append(self.y[j-1])
                j -=1
        
            elif traceback_matrix[i][j]== '^':
                x_prime.append(self.x[i-1])
                y_prime.append('-')
                i -=1  
        return (''.join(x_prime[::-1]), ''.join(y_prime[::-1]))#reverse and append the strings to get the optimal alignments for both sequences


def main(): #main function to run the program
    x,y= parse_FASTA_file(sys.stdin) #add the standin for the FASTA files
    align_1 = GlobalSeqAlignment(x,y, 4,-2,-2) #initialize the alignmnet class with the sequences and the scoring system
    x_prime, y_prime =align_1.get_optimal_alignment() #get the optimal alignment for the sequences
    print(x_prime)
    print(y_prime)  #return the optimal alignment for the sequences

if __name__ == '__main__': #run the main function oin comand console
    main()
