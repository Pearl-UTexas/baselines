#!/usr/bin/env python
from OpenGL import GLU
import os, sys
import gym, roboschool
import argparse
import os
from tqdm import tqdm

sys.path.append('../../')
from baselines import bench, logger
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def test(env_id, num_timesteps, seed, model_path, target=0, num_iter=100):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy, CnnPolicy
    import gym
    from gym import wrappers
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    with tf.Graph().as_default():
        ncpu = 1
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        with sess.as_default():
            def make_env():
                import numpy as np
                import colorsys
                COLOR_SET = [ tuple(int(c*255) for c in colorsys.hsv_to_rgb(h/360.,1,1))
                            for h in range(0,360,20) ]

                np.random.seed(0)
                np.random.shuffle(COLOR_SET)
                COLOR_SET = COLOR_SET[:4]

                env = gym.make(env_id)
                env.unwrapped.set_goals( [target] )
                env.unwrapped.set_targets_color( COLOR_SET )
                #env.unwrapped.set_fixed(1)
                #env = wrappers.Monitor(env,os.path.join('/tmp/videos/'),force=True)
                return env
            env = DummyVecEnv([make_env],render=False)
            #env = DummyVecEnv([make_env],render=True)
            env = VecNormalize(env,True,True)
            #env = VecNormalize(env,False,False)
            env.load(model_path)

            #set_global_seeds(seed)
            policy = MlpPolicy
            #policy = CnnPolicy

            success = 0
            for states,actions,images in \
                    tqdm(
                        ppo2.test(policy=policy, env=env, nsteps=num_timesteps, model_dir=model_path, num_iter=num_iter),
                        total=num_iter):
                #print(states.shape, actions.shape, images.shape)

                if( len(states) < 150):
                    success+=1

                #import moviepy.editor as mpy
                #clip = mpy.ImageSequenceClip(list(images),fps=30)
                #clip.write_videofile('video.mp4', verbose=False,ffmpeg_params=['-y'],progress_bar=False)
                #input()

            print('Success Rate: %f(%d/%d)'%(1.*success/num_iter,success,num_iter))

        import numpy as np
        np.savez('test.npz', states=states,actions=actions,images=images)
        sess.close()
        return success

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    #parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--env', help='environment ID', default='RoboschoolReacher-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2048))
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()

    """
    list_of_expr=[
        #('gt_reward',1,'/tmp/openai-2018-01-21-21-36-45-711300/checkpoints/',[]),
        #('gt_reward',0,'/tmp/openai-2018-01-21-21-33-23-601570/checkpoints/',[]),
        #('maml',0,'/tmp/openai-2018-01-20-21-13-09-650082/checkpoints/',[]),
        #('maml',1,'/tmp/openai-2018-01-20-17-58-57-139601/checkpoints/',[]),
        #('perfect',0,'/tmp/openai-2018-01-20-20-39-15-859841/checkpoints/',[]),
        #('perfect',1,'/tmp/openai-2018-01-20-17-47-06-661280/checkpoints/',[])
        ('one_demo',0,'/tmp/openai-2018-09-09-12-11-34-642682/checkpoints/',[]),
    ]

    for x in range(10,490,10):
        for name,target,model_dir,results in list_of_expr:
            if( not os.path.isfile('%s%05d.env_stat.pkl'%(model_dir,x)) ):
                continue

            success_cnt = \
                test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                     model_path='%s%05d'%(model_dir,x),
                     target=target,
                     num_iter=100)
            results.append( (x,success_cnt) )

    for name,target,_,results in list_of_expr:
        print(name,target)
        print(name, ','.join([str(val) for _,val in results]))
    """

    test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,model_path=args.model_path)

    #model_path='/tmp/openai-2018-01-21-21-36-45-711300/checkpoints/00250') #With GroundTruth Reward Function(subtask 1)
    #model_path='/tmp/openai-2018-01-21-21-33-23-601570/checkpoints/00280') #With GroundTruth Reward Function(subtask 0) 288/300
    #model_path='/tmp/openai-2018-01-20-21-13-09-650082/checkpoints/00240') #MAML Aligner(subtask 0)      256/300
    #model_path='/tmp/openai-2018-01-20-17-58-57-139601/checkpoints/00480') #MAML Aligner(subtask 0)     125/300
    #model_path='/tmp/openai-2018-01-20-20-39-15-859841/checkpoints/00200') # Pefect Aligner(subtask 0)  236/300
    #model_path='/tmp/openai-2018-01-20-17-47-06-661280/checkpoints/00480') # Perfect Aligner(subtask 1) 249/300
    #model_path='/tmp/openai-2018-01-20-15-12-34-106221/checkpoints/00290') #Maml ALigner(task: 1), new  79/300
    #model_path='/tmp/openai-2018-01-20-21-40-09-420115/checkpoints/00480') #Maml Aligner(task:1),
    #model_path='/tmp/openai-2018-01-20-00-18-36-180244/checkpoints/00480') # Whole Video(Unordered) Not working.
    #model_path='/tmp/openai-2018-01-20-00-17-41-806118/checkpoints/00200') # Whole Video(Ordered) Not working.
    #model_path='/tmp/openai-2018-01-19-19-13-58-892638/checkpoints/00130') #00200 #00300 Maml Aligner ( target: 1)
    #model_path='/tmp/openai-2018-01-19-17-28-45-890769/checkpoints/00290') #Maml Aligner ( target:0 )
    #model_path='/tmp/openai-2018-01-18-23-15-38-650901/checkpoints/00150') #Perfect Aligner ( target: 0)
    #model_path='/tmp/openai-2018-01-18-23-01-48-386512/checkpoints/00050')
    #model_path='/tmp/openai-2018-01-18-21-15-14-733256/checkpoints/00050')
    #model_path='/tmp/openai-2018-01-18-20-58-46-504336/checkpoints/00130')
    #model_path='/tmp/openai-2018-01-17-21-14-33-709102/checkpoints/00340')
    #model_path='/tmp/openai-2018-01-16-21-13-42-297830/checkpoints/00100')
    #model_path='/tmp/openai-2018-01-16-20-46-52-719658/checkpoints/00220')
    #model_path='/tmp/openai-2018-01-16-20-27-25-798771/checkpoints/00150')
    #model_path='/tmp/openai-2018-01-16-20-19-14-061814/checkpoints/00110')
    #model_path='/tmp/openai-2018-01-16-20-08-11-760768/checkpoints/00230')
    #model_path='/tmp/openai-2018-01-16-20-08-11-760768/checkpoints/00230')
    #model_path='/tmp/openai-2018-01-16-17-09-00-358335/checkpoints/00140')
    #model_path='/tmp/openai-2018-01-16-16-41-53-640746/checkpoints/00160')
    #model_path='/tmp/openai-2018-01-15-16-04-04-089676/checkpoints/00230')
    #model_path='/tmp/openai-2018-01-15-15-49-32-706043/checkpoints/00170')
    #model_path='/tmp/openai-2017-12-18-18-05-34-607331/checkpoints/00020')
    #model_path='/tmp/openai-2017-12-18-17-38-55-610551/checkpoints/00060')

if __name__ == '__main__':
    main()
