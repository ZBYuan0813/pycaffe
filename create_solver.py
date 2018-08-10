# -*- coding: utf-8 -*-
class CaffeSolver:
	'''
	This class is generate solver.prototxt
	'''
	def __init__(self,args):
		self.sp = {}
		self.sp['train_net'] = args.train_net                    # 训练配置文件
		if args.test_net != '':
			self.sp['test_net'] = args.test_net                      # 测试配置文件
			self.sp['test_iter'] = args.test_iter                    # 测试迭代次数
			self.sp['test_interval'] = args.test_interval            # 测试间隔
		self.sp['base_lr'] = args.base_lr                        # 基础学习率
		self.sp['display'] = args.display                        # 屏幕日志显示间隔
		self.sp['max_iter'] = args.max_iter                      # 最大迭代次数
		self.sp['lr_policy'] = args.lr_policy                    # 学习率变化规律
		self.sp['gamma'] = args.gamma                            # 学习率变化指数
		self.sp['momentum'] = args.momentum                      # 动量
		self.sp['weight_decay'] = args.weight_decay              # 权值衰减
		self.sp['stepsize'] = args.stepsize                      # 学习率变化频率
		self.sp['snapshot'] = args.snapshot                      # 保存model间隔
		self.sp['snapshot_prefix'] = args.snapshot_prefix        # 保存的model前缀
		self.sp['solver_mode'] = args.solver_mode                # 是否使用gpu
		self.sp['solver_type'] = args.solver_type                # 优化算法
		self.sp['device_id'] = args.device_id

	def write_solver(self):
	#写入文件
		solver_file="solver/solver.prototxt"
		with open(solver_file, 'w') as f:
			for key, value in sorted(self.sp.items()):
				if not(type(value) is str):
					value = str(value)
				f.write('%s: %s\n' % (key, value))
		return solver_file


'''
def solver(args):
	#solver_file=path+'solver.prototxt'               #solver文件保存位置
	sp = {}
	sp['train_net'] = args.train_net                    # 训练配置文件
	if args.test_net != '':
		sp['test_net'] = args.test_net                      # 测试配置文件
		sp['test_iter'] = args.test_iter                    # 测试迭代次数
		sp['test_interval'] = args.test_interval            # 测试间隔
	sp['base_lr'] = args.base_lr                        # 基础学习率
	sp['display'] = args.display                        # 屏幕日志显示间隔
	sp['max_iter'] = args.max_iter                      # 最大迭代次数
	sp['lr_policy'] = args.lr_policy                    # 学习率变化规律
	sp['gamma'] = args.gamma                            # 学习率变化指数
	sp['momentum'] = args.momentum                      # 动量
	sp['weight_decay'] = args.weight_decay              # 权值衰减
	sp['stepsize'] = args.stepsize                      # 学习率变化频率
	sp['snapshot'] = args.snapshot                      # 保存model间隔
	sp['snapshot_prefix'] = args.snapshot_prefix        # 保存的model前缀
	sp['solver_mode'] = args.solver_mode                # 是否使用gpu
	sp['solver_type'] = args.solver_type                # 优化算法
	return sp

def write_solver(args):
    #写入文件
    sp = solver(args)
    solver_file="solver.prototxt"
    with open(solver_file, 'w') as f:
        for key, value in sorted(sp.items()):
            if not(type(value) is str):
            	value = str(value)
                #raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
    return solver_file
'''
if __name__ == '__main__':
    write_solver()
