To run in jupter notebook on Wendian

## For dmg framework:
1. I have moved my environments saved on wendian due to space limit, therefore make sure you activate the right environment from the right location, use:

conda activate dmg312

2. Now go to your delta model location, which is at /u/au/ac/zhennanshi/znprojects/DM/generic_deltamodel, then run the toml file at where the toml is generic_deltamodel/pyproject.toml, in order to install all the dependencies:

python -m pip install -e .  ### Do not use pip install -e .###

3. Create the kernel for the jupter notebook to run, skip if the kenerl is already created!

python -m pip install ipykernel  # if installed already, skip this line
python -m ipykernel install --user --name dmg312 --display-name "Python (dmg312_renewed)"

4. Now you can select on the Jupter notebook: Kenrnel --> Reconnect to (select the one you created), skip if you used the same kernel as last time

## For hydrodl2:
Since you also used hydrodl2, which is at https://github.com/mhpi/hydrodl2, make sure it is up-to-date:

0. Back-up files you modified
download a version locally

1. You want to keep the HBV class that you modified:
git stash

2. You want to pull
git pull

3. Check what is changed
git stash pop

4. Go to where the toml file is at in hydrodl2, then run:
python -m pip install -e .

5. you can check in dmg to see which hydrodl2 it is using:
import hydrodl2
print(hydrodl2.__file__)

## For GEFS Data

1. Make sure it is at /u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII/, eg. /u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII/ens01 for ensemble 1

2. To be updated...



Some useful commands:
1. search a string for debugging purpose

grep -R "your_string" folder_name

2. check the training progress by running the file or in terminal
!tail -n 20 -f train_progress.log
module load apps/python3
conda activate hydro

3. Copy the file on wendian, eg. /beegfs/scratch/zhennanshi/DM/ and save it locally
scp -r zhennanshi@wendian.mines.edu:/beegfs/scratch/zhennanshi/DM/ ~/Downloads/

4. Use git to save your files
    4.0 The current DM on wendian is a git repository, a main branch of my **fork** to the master generic_deltamodel, which should be at: git@github.com:ZhennanShi1/generic_deltamodel.git, on the main branch
        you can check by git remote -v, the outputs should be:
        origin  git@github.com:ZhennanShi1/generic_deltamodel.git (fetch)
        origin  git@github.com:ZhennanShi1/generic_deltamodel.git (push)
        Note: if it shows: 
        origin  https://github.com/ZhennanShi1/generic_deltamodel.git (fetch)
        origin  https://github.com/ZhennanShi1/generic_deltamodel.git (push)
        it means that you used http, instead of ssh, this means you might need to enter token everytime, which i suggest use git ssh:
        here is how you can do that:
        4.0.1 git remote set-url origin git@github.com:ZhennanShi1/generic_deltamodel.git
        4.0.2 now verify with command: git remote -v, you should see:
            origin  git@github.com:ZhennanShi1/generic_deltamodel.git
        4.0.3 Check your SSH key: ls ~/.ssh, If you have
            id_rsa
            id_rsa.pub
            then it is good
        4.0.4 Show the public key: cat ~/.ssh/id_rsa.pub
        4.0.5 Copy the entire line and add it to GitHub: GitHub → Settings → SSH and GPG keys → New SSH Key
        4.0.6 Now run this test in terminal: ssh -T git@github.com, you should see something like
            Hi ZhennanShi1! You've successfully authenticated...
    4.1 save your current work locally first: scp -r zhennanshi@wendian.mines.edu:/beegfs/scratch/zhennanshi/DM/ ~/Downloads/
    4.2 save to your current work but not push yet: 
        git add .
        git commit -m "Save local work before syncing upstream"
    4.3 Push to my branch 
        git push origin main
        **you may see errors such as big files on wendian cannot be pushed to your branch**, you can:
        4.3.1 git merge --abort    ###abort the merge first
        4.3.2 git status           ###check the current status
        4.3.3 git rm --cached example/hydrology/predownloaded/camels_dataset ###remove the big files
        4.3.4 igonore all the big files that are uploaded:
            4.3.1 find the gitignore file: ls -la  ###the gitignore should be at generic_deltamodel/.gitignore 
            4.3.2 open the gitignore file using nano: nano .gitignore
            4.3.3 add eg. example/hydrology/predownloaded/  at the end of the gitignore file
            4.3.4 save the modified gitignore file: ctrl+O, ENTER, then ctrl+X
            4.3.5 add the gitignore file: git add .gitignore, then git commit --amend --no-edit
            4.3.6 try push to the origin main again: git push origin main --force-with-lease
    4.4 Now check everything is pushed to your branch, make sure it is clean:
        git status
        git log --oneline -3
        ###if not clean, repeat 4.2 and 4.3
    4.5 Sync with upstream (the master generic_deltamodel you see on github)
        git remote add upstream git@github.com:mhpi/generic_deltamodel.git ###if you see error: remote upstream already exists. that is fine!
        git fetch upstream ###nothing will show up
        git checkout main ###if you see Already on 'main', Your branch is up to date with 'origin/main. This is very good!
        git merge upstream/master
        git push origin main
    4.6 If you sync with upstream, most likely that you will need to run "python -m pip install -e ." at where the toml file is.
        
        
