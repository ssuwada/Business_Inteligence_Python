

In each folder are stored different things.


1. crossing_raw_data -> 	All data that I used to create this raport and perform calculations. 

2. Generated_excel ->	Two excel files, one showing what angle is exactly from calculation for each file, second file showing  
Files that are divided into groups of theoretical angles that they belong.  Deviation angle in this case is subtract calculated value of angle from theoretical value.\


3. Pictures 	->	In this folder you can find every picture that was used in raport or even was not here but can be helpful for understanding the problem. 


4. Code  ->	Folder that include every code that was written by me.

crossing_project.py   ->	first of all check if path for loading data files is correct(read_all_csv() function), then you can play this code. It will generate you 2 excel files that were described in 2 point.
It will also create deviation angle file and create segregated files depend on angle, for example:
'nameFile_90.txt' - will only include files that they theoretical value is 90 degree. 


velocity_pedestrians.py ->	As an output it will create \'93pos_0.txt\'94  that will include every value for every file from all csv files provided for every agent about velocity at given time. 

deviation_angle.py -> 	Do the same as velocity but calculate deviation angle. First calculate main angle line between first positions and last position of every agent. Then its performed for every step in time and its deviation angle is subtract of this time agent deviation and overall agent deviation, which gives us deviation angle at every time. It will create files 'deviat_0_full.txt'


plots_hist.py 	->	Plots using generated files .txt histogram for deviation angle and velocity. 


5. txt_files -> all txt files that I generated for this specific data. 


