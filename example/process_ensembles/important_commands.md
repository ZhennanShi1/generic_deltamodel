When you run it, make sure you pull respective files out of this old_example folder

some common command that will be useful
# check the status of the submitted job
squeue -u $USER

# cancel the job submitted
scancel -u $USER

# check status of the nodes
sinfo  # more general
sinfo -N -l # more detailed

# submit job to wendian
sbatch filename


other commands

# check the last 40 lines of a file
tail -n 40 filename

# kill vscode command, if you see error message like "-bash: fork: retry:..."
pkill -u $USER -f '\.vscode-server'

# calculate the folder size, if you cannot open vscode, it could be out of quota size 
du -sh ~/* ~/.??* 2>/dev/null | sort -h

# check the nc file header
ncdump -h "$FILE" | head -100
# if you want to see all the header, use  without | head -100

# the header of the ens01 filedimensions:
dimensions:
        catchment-id = 54667 ;
        time = 80 ;