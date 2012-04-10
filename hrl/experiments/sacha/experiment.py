### Imports ###

# Package Imports
from hrl import HRL

# Qualified Imports
import numpy as np
import argparse as ap
import sys
import os

# Unqualified Imports
from random import uniform


### Argument Parser ###


prsr = ap.ArgumentParser(description= "This is an experiment. It takes an argument in the form of a subject name, which it will use to name the result file. The experiment involves asking subjects to set a central circle to the luminance in between to other circles. Fine and coarse controls are provided by directional keys, and the selected luminance is entered with space/white.")

prsr.add_argument('sbj',default='subject',nargs='?',help="The name of the subject of the experiment. Default: subject")


### Main ###


def main():

    # Parse args
    args = prsr.parse_args()

    # HRL parameters
    wdth = 1024
    hght = 768

    # Section the screen - used by Core Loop
    wqtr = wdth/4.0
    
    # IO Stuff
    dpxBool = False
    dfl = 'design.csv'
    rfl = 'results/' + args.sbj + '.csv'
    flds = ['TrialTime','SelectedLuminance']
    btns = ['Yellow','Red','Blue','Green','White']

    # Central Coordinates (the origin of the graphics buffers is at the centre of the
    # screen. Change this if you don't want a central coordinate system. If you delete
    # this part the default will be a matrix style coordinate system.
    coords = (-0.5,0.5,-0.5,0.5)
    flipcoords = False

    # Pass this to HRL if we want to use gamma correction.
    lut = 'LUT.txt'
    # If fs is true, we must provide a way to exit with e.g. checkEscape().
    fs = False

    # Step sizes for luminance changes
    smlstp = 0.01
    bgstp = 0.1

    # HRL Init
    hrl = HRL(wdth,hght,dpx=dpxBool,dfl=dfl,rfl=rfl,rhds=flds
              ,btns=btns,fs=fs,coords=coords,flipcoords=flipcoords)

    # Core Loop
    for dsgn in hrl.dmtx:

        # Load Trial
        mnl = float(dsgn['MinLuminance'])
        mxl = float(dsgn['MaxLuminance'])
        rds = float(dsgn['Radius'])

        # Create Patches
        mnptch = hrl.newTexture(np.array([[mnl]]),'circle')
        mxptch = hrl.newTexture(np.array([[mxl]]),'circle')
        cntrllm = uniform(0.0,1.0)
        cntrlptch = hrl.newTexture(np.array([[cntrllm]]),'circle')

        # Draw Patches
        mnptch.draw((-wqtr,0),(2*rds,2*rds))
        mxptch.draw((wqtr,0),(2*rds,2*rds))
        cntrlptch.draw((0,0),(2*rds,2*rds))
        # Draw but don't clear the back buffer
        hrl.flip(clr=False)

        # Prepare Core Loop logic
        btn = None
        t = 0.0
        escp = False

        # Adjust central patch
        while ((btn != 'White') & (escp != True)):
            
            (btn,t1) = hrl.readButton()
            t += t1

            if btn == 'Yellow':
                cntrllm += smlstp
            elif btn == 'Red':
                cntrllm += bgstp
            elif btn == 'Blue':
                cntrllm -= smlstp
            elif btn == 'Green':
                cntrllm -= bgstp

            # Bound Checking
            if cntrllm > 1: cntrllm = 1
            if cntrllm < 0: cntrllm = 0

            # Update display
            cntrlptch = hrl.newTexture(np.array([[cntrllm]]),'circle')
            cntrlptch.draw((0,0),(2*rds,2*rds))
            hrl.flip(clr=False)

            if hrl.checkEscape(): escp = True

        # Save results of trial
        hrl.rmtx['TrialTime'] = t
        hrl.rmtx['SelectedLuminance'] = cntrllm
        hrl.writeResultLine()
        
        # Check if escape has been pressed
        if escp: break

    # Experiment is over!
    hrl.close()


### Run Main ###


if __name__ == '__main__':
    main()
