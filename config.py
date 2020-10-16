class config:
    # 'origin', 'improve'
    method = 'improve'

    # 'heart', 'cleveland', 'dermatology', 'ionosphere', 'wine', 'vehicle', 'sonar', 'glass', 'segmentation', 'hepatitis'
    data = 'sonar'

    # 1NN, 3NN, 5NN, SVM, C4.5
    classifier = '5NN'

    life_time = 15
    area_limit = 50
    transfer_rate = 0.05

    # 10-fold, 2-fold, 7-3
    fitness_type = '7-3'

    # usually 10-50
    epoch = 15
