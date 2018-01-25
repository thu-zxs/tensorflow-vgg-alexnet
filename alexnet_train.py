from utils import DigitDataset
from alexnet import ALEXnet
import tensorflow as tf
import time
import sys
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batchSize', 24,
                            """Specify batch size""")
tf.app.flags.DEFINE_integer('maxIter', 10000,
                            """Specify maximum iteration""")
tf.app.flags.DEFINE_float('trainRatio', 0.8,
                            """Ratio of training samples""")
tf.app.flags.DEFINE_float('weightDecay', 1.0,
                            """weigh decay""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                            """specify learning rate""")
tf.app.flags.DEFINE_boolean('isVal', False,
                            """validation mode or not""")
tf.app.flags.DEFINE_boolean('isTest', False,
                            """test mode or not""")
tf.app.flags.DEFINE_string('modelPath', "",
                            """model path where model exists""")


def main(_):

    time_format = '%Y%m%d_%H%M%S'
    time_post = time.strftime(time_format, time.localtime())
    
    mode = 'train' if not FLAGS.isTest else 'test'
    D = DigitDataset('.', trainRatio=FLAGS.trainRatio, h=224, w=224, labelNum=10, mode=mode)

    img = tf.placeholder(tf.float32, [None, 224, 224, 1])
    target = tf.placeholder(tf.float32, [None, D.labelNum])
    keep_prob = tf.placeholder(tf.float32, None)
    batch_size = FLAGS.batchSize
    iters_per_epoch = int(float(D.trainData.shape[0]) / batch_size) if not FLAGS.isTest else None

    model = ALEXnet(
        input_image=img,
        target=target,
        init_lr=FLAGS.lr,
        keep_prob=keep_prob,
        wd=FLAGS.weightDecay
        #rgb_mean=np.array([0,0,0],dtype=np.float32),
    ) 
    plus1 = tf.assign(model.global_step, model.global_step+1)

    test_accuracy = tf.placeholder(tf.float32, None)
    test_accuracy_summary = tf.summary.scalar('test_accuracy', test_accuracy)
    test_summary = tf.summary.merge([test_accuracy_summary])

    init_op = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(
        device_count = {'GPU': int(1)}
    )
        
    saver = tf.train.Saver()

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        print("uninitialized variables: \n{}".format(sess.run(tf.report_uninitialized_variables())))
        #saver.restore(sess, 'log_dir/img_log-20161225_124305/6th_vgg_iter_4000.ckpt')

        if not FLAGS.isVal and not FLAGS.isTest:
            summary_writer = tf.summary.FileWriter('log_dir/alex_log-{}'.format(time_post), sess.graph)

            """ Writing a metafile for embedding visulization
            """
            os.mkdir('log_dir/alex_log-{}/embedding'.format(time_post))
            with open('log_dir/alex_log-{}/embedding/metadata.tsv'.format(time_post), 'w') as metafile:
                #metafile.write('Class\n')
                labels_str = np.array(D.trainLabel).astype(np.str)
                #to_write = map('\t'.join, zip(D.get_feature['training']['files'], labels_str))
                to_write = labels_str.tolist()
                to_write = '\n'.join(to_write)
                metafile.write(to_write)
            
            for ind, (datas, _1hot_labels) in enumerate(D.genMinibatch(batch_size, D.trainData, D.trainLabel)):
                sess.run(model.optimize, feed_dict={img:datas, target:_1hot_labels, keep_prob:0.5})
                sess.run(plus1)
                if (ind+1) % 5 == 0:
                    print('[----iteration: {}, lr: {}----]\nloss:{},\naccuracy:{}\n'.format(
                        ind + 1,
                        sess.run(model.lr),
                        sess.run(model.cross_entropy_loss, feed_dict={img:datas, target:_1hot_labels, keep_prob:0.5}),
                        sess.run(model.accuracy, feed_dict={img:datas, target:_1hot_labels, keep_prob:1.0})
                    ))
                    #print(sess.run(model.LSTM, feed_dict={sound:datas, target:_1hot_labels}))
                    summary_str = sess.run(model.summary, feed_dict={img:datas, target:_1hot_labels, keep_prob:1.0})
                    summary_writer.add_summary(summary_str, ind+1)
                    summary_writer.flush()

                """ A validation epoch right after training
                    And a model saver.
                    Plus an embeddings visualizer.
                """
                if (ind+1) % (1000) == 0:
                    saver.save(sess, 'log_dir/alex_log-{}/1st_iter_{}.ckpt'.format(time_post,ind+1))
                    correct_count = 0
                    for idx, (datas1, _1hot_labels1) in enumerate(D.genMinibatch(20, D.valData, D.valLabel)):
                        correct_count += sess.run(tf.reduce_sum(tf.cast(model.corrects, tf.float32)), feed_dict={img:datas1, target:_1hot_labels1, keep_prob:1.0})
                        if idx == 9:
                            break
                    print("Total Test Accuracy: {}".format(correct_count/200))
                    test_summary_str = sess.run(test_summary, feed_dict={img:datas1, target:_1hot_labels1, keep_prob:1.0, test_accuracy:correct_count/200})
                    summary_writer.add_summary(test_summary_str, ind+1)
                    summary_writer.flush()

                if ind+1 == FLAGS.maxIter:
                    break


        elif FLAGS.isVal:

            correct_count = 0
            valSize = D.valData.shape[0]
            batch = 20
            saver.restore(sess, FLAGS.modelPath)
            for idx, (datas, _1hot_labels) in enumerate(D.genMinibatch(batch, D.valData, D.valLabel)):
                correct_count += sess.run(tf.reduce_sum(tf.cast(model.corrects, tf.float32)), feed_dict={img:datas, target:_1hot_labels, keep_prob:1.0})
                if idx+1 == valSize/batch:
                    break
            print("total validation accuracy: {}".format(correct_count/valSize))
            print('validation set size: {}'.format(valSize))

        elif FLAGS.isTest:

            testSize = D.data.shape[0]
            batch = 20
            saver.restore(sess, FLAGS.modelPath)
            for idx, (datas, _) in enumerate(D.genMinibatch(batch, D.data)):
                if idx == 0:
                    res = sess.run(tf.argmax(model.predict,1), feed_dict={img:datas, keep_prob:1.0})
                else:
                    res = np.concatenate((res, sess.run(tf.argmax(model.predict,1), feed_dict={img:datas, keep_prob:1.0})))
                    if idx+1 == testSize/batch:
                        break
            with open('submission.csv','w') as f:
                f.write('ImageId,Label\n')
                for i,p in enumerate(res.tolist()):
                    f.write("{},{}\n".format(i+1,p))
            




if __name__=='__main__':
    tf.app.run()
