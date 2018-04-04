
from incremental_svd_tester import IncrementalSVDTester
from incremental_svd_trainer import IncrementalSVDTrainer

svd_trainer = IncrementalSVDTrainer(
        '../data/20k-users/training_movies.csv',
        '../data/20k-users/training_ratings.csv',
        '../data/20k-users/training_links.csv',
    )
svd_trainer.configure(0.1, 0.15, 8)

print 'Before function optimization:'
print 'Training RMSE: %s' % svd_trainer.training_rmse
print 'CV RMSE: %s\n' % svd_trainer.cross_validation_rmse

svd_trainer.batch_gradient_descent()

print 'After function optimization:'
print 'Training RMSE: %s' % svd_trainer.training_rmse
print 'CV RMSE: %s\n' % svd_trainer.cross_validation_rmse

svd_trainer.export_feature('../data/20k-users')

svd_tester = IncrementalSVDTester(
        '../data/1k-users/training_ratings.csv',
        svd_trainer.movies
    )
svd_tester.configure(0.1, 0.15, 8)

print 'Before function optimization:'
print 'Training RMSE: %s' % svd_tester.training_rmse
print 'Test RMSE: %s\n' % svd_tester.test_rmse

svd_tester.content_based_batch_gradient_descent()

print 'After function optimization:'
print 'Training RMSE: %s' % svd_tester.training_rmse
print 'Test RMSE: %s\n' % svd_tester.test_rmse
