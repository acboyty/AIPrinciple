class config:
    # 'origin', 'improve'
    method = 'origin'

    # 'heart', 'cleveland', 'dermatology', 'ionosphere', 'wine', 'vehicle', 'sonar', 'glass', 'segmentation', 'hepatitis'
    data = 'sonar'

    # 1NN, 3NN, 5NN, SVM, C4.5
    classifier = 'SVM'

    life_time = 15
    area_limit = 50
    transfer_rate = 0.05

    # 10-fold, 2-fold, 7-3
    fitness_type = '10-fold'

    # usually 10-50
    epoch = 20
