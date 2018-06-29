#xgboost.train()API
xgboost.train(params,dtrain,num_boost_round=10,evals=(),obj=None,feval=None,maximize=False,early_stopping_rounds=None,
evals_result=None,verbose_eval=True,learning_rates=None,xgb_model=None)

"""
params 这是一个字典，里面包含着训练中的参数关键字和对应的值，形式是params = {‘booster’:’gbtree’,’eta’:0.1}
dtrain 训练的数据
num_boost_round 这是指提升迭代的个数
evals 这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals = [(dtrain,’train’),(dval,’val’)]或者是evals = [(dtrain,’train’)],对于第一种情况，它使得我们可以在训练过程中观察验证集的效果。
obj,自定义目的函数
feval,自定义评估函数
maximize ,是否对评估函数进行最大化
early_stopping_rounds,早期停止次数 ，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。这要求evals 里至少有 一个元素，如果有多个，按最后一个去执行。返回的是最后的迭代次数（不是最好的）。如果early_stopping_rounds 存在，则模型会生成三个属性，bst.best_score,bst.best_iteration,和bst.best_ntree_limit
evals_result 字典，存储在watchlist 中的元素的评估结果。
verbose_eval (可以输入布尔型或数值型)，也要求evals 里至少有 一个元素。如果为True ,则对evals中元素的评估结果会输出在结果中；如果输入数字，假设为5，则每隔5个迭代输出一次。
learning_rates 每一次提升的学习率的列表，
xgb_model ,在训练之前用于加载的xgb model。
"""

# 首先 parameters 设置如下：
params = {
            'booster':'gbtree',
            'objective':'binary:logistic',
            'eta':0.1,
            'max_depth':10,
            'subsample':1.0,
            'min_child_weight':5,
            'colsample_bytree':0.2,
            'scale_pos_weight':0.1,
            'eval_metric':'auc',
            'gamma':0.2,            
            'lambda':300
}

"""
colsample_bytree 要依据特征个数来判断
objective 目标函数的选择要根据问题确定，
如果是回归问题 ，一般是 reg:linear , reg:logistic , count:poisson 
如果是分类问题，一般是binary:logistic ,rank:pairwise

参数初步定之后划分20%为验证集，准备一个watchlist 给train和validation set ,
设置num_round 足够大（比如100000），以至于你能发现每一个round 的验证集预测结果，
如果在某一个round后 validation set 的预测误差上升了，你就可以停止掉正在运行的程序了。
"""

watchlist = [(dtrain,'train'),(dval,'val')]
model = xgb.train(params,dtrain,num_boost_round=100000,evals = watchlist)

1. 首先调整max_depth ,通常max_depth 这个参数与其他参数关系不大，初始值设置为10，
找到一个最好的误差值，然后就可以调整参数与这个误差值进行对比。比如调整到8，
如果此时最好的误差变高了，那么下次就调整到12；如果调整到12,误差值比10 的低，
那么下次可以尝试调整到15.


2. 在找到了最优的max_depth之后，可以开始调整subsample,初始值设置为1，然后调整到0.8 
如果误差值变高，下次就调整到0.9，如果还是变高，就保持为1.0


3. 接着开始调整min_child_weight , 方法与上面同理


4. 再接着调整colsample_bytree


5. 经过上面的调整，已经得到了一组参数，这时调整eta 到0.05，
然后让程序运行来得到一个最佳的num_round,(在 误差值开始上升趋势的时候为最佳 )