import os
import glob
import shutil
import argparse

def main(expdir):
    '''read the results for an experiment on a database'''

    if not os.path.isdir(expdir):
        raise Exception('cannot find expdir: %s' % expdir)

    #get a list of speaker directories
    exps = [e for e in os.listdir(expdir)
            if os.path.isdir(os.path.join(expdir, e))]   
    
    for exp in exps:
        
        exprootdir = os.path.join(expdir, exp)
        
        speakers = [s for s in os.listdir(exprootdir)
                   if os.path.isdir(os.path.join(exprootdir, s))]
        
        for speaker in speakers:
            
            exppath = os.path.join(exprootdir, speaker)

        #exp_path = os.path.join('/esat/spchtemp/scratch/pwang/pre-training/capsule/fluent_results/5', speaker)

            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'attention')):
                shutil.rmtree(f, ignore_errors=True)
        #for f in glob.glob(
                #'%s*' % os.path.join(exppath, 'save_model')):
            #shutil.rmtree(f, ignore_errors=True)
        #for f in glob.glob(
                #'%s*' % os.path.join(exppath, 'model')):
            #shutil.rmtree(f, ignore_errors=True)
            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'outputs')):
                shutil.rmtree(f, ignore_errors=True)
            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'logdir')):
                shutil.rmtree(f, ignore_errors=True)
            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'logdir-decode')):
                shutil.rmtree(f, ignore_errors=True)
            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'model')):
                shutil.rmtree(f, ignore_errors=True)

            for f in glob.glob(
                    '%s*' % os.path.join(exppath, 'weights')):
                shutil.rmtree(f, ignore_errors=True)



if __name__ == '__main__':
    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')

    args = parser.parse_args()

    main(args.expdir)
#path = "/esat/spchdisk/scratch/pwang/before_511/RCCN_encode_9/4blocks_exp4/"
