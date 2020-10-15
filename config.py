class config:
    # 'heart', 'cleveland', 'dermatology', 'ionosphere', 'wine', 'vehicle', 'sonar', 'glass', 'segmentation', 'hepatitis'
    data = 'segmentation'

    # 1NN, 3NN, 5NN, SVM, C4.5
    classifier = '3NN'

    life_time = 15
    area_limit = 50
    transfer_rate = 0.05

    # 10-fold, 2-fold, 7-3
    fitness_type = '7-3'

    # usually 10-50
    epoch = 15
