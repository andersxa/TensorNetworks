#%%
import pandas as pd

mnist_csv = './results/Mnistres.csv'
fashion_mnist_csv = './results/FMnistres.csv'

mnist_df = pd.read_csv(mnist_csv)

# Add results:
# MNIST:
# Rank: 2
# Params: 47110
# Test Acc: 92.790000
# ------------------
# Rank: 5
# Params: 117760
# Test Acc: 95.300000
# ------------------
# Rank: 10
# Params: 235510
# Test Acc: 96.260000
# ------------------
# Rank: 20
# Params: 471010
# Test Acc: 96.920000
# ------------------
# Rank: 50
# Params: 1177510
# Test Acc: 97.440000
# ------------------
# Rank: 150
# Params: 3532510
# Test Acc: 97.770000
# ------------------

cpd_mnist = [
    ('CPD', 2, 47110, 92.79),
    ('CPD', 5, 117760, 95.30),
    ('CPD', 10, 235510, 96.26),
    ('CPD', 20, 471010, 96.92),
    ('CPD', 50, 1177510, 97.44),
    ('CPD', 150, 3532510, 97.77),
]

# Fashion MNIST:
# Rank: 2
# Params: 47110
# Test Acc: 84.590000
# ------------------
# Rank: 5
# Params: 117760
# Test Acc: 86.010000
# ------------------
# Rank: 10
# Params: 235510
# Test Acc: 86.520000
# ------------------
# Rank: 20
# Params: 471010
# Test Acc: 87.070000
# ------------------
# Rank: 50
# Params: 1177510
# Test Acc: 87.400000
# ------------------
# Rank: 150
# Params: 3532510
# Test Acc: 87.920000
# ------------------

fashion_cpd = [
    ('CPD', 2, 47110, 84.59),
    ('CPD', 5, 117760, 86.01),
    ('CPD', 10, 235510, 86.52),
    ('CPD', 20, 471010, 87.07),
    ('CPD', 50, 1177510, 87.40),
    ('CPD', 150, 3532510, 87.92),
]

# MNIST TNML:
# 155720, 0.9815
# 621192, 0.9755
# 2478280, 0.9835
# 9887432, 0.9814
tnml_mnist = [
    ('TNML', 10, 155720, 98.15),
    ('TNML', 20, 621192, 97.55),
    ('TNML', 40, 2478280, 98.35),
    ('TNML', 80, 9887432, 98.14),
]

# Fashion MNIST TNML:
# 155720, 0.8749
# 621192, 0.8785
# 2478280, 0.8800
# 9887432, 0.8749
tnml_fashion = [
    ('TNML', 10, 155720, 87.49),
    ('TNML', 20, 621192, 87.85),
    ('TNML', 40, 2478280, 88.00),
    ('TNML', 80, 9887432, 87.49),
]

fashion_mnist_df = pd.read_csv(fashion_mnist_csv)

# Drop rows with "type_II" in model_type
mnist_df = mnist_df[~mnist_df['model_type'].str.contains('type_II')]
fashion_mnist_df = fashion_mnist_df[~fashion_mnist_df['model_type'].str.contains('type_II')]

# Drop rows with `std_test_accuracy` > 1
mnist_df = mnist_df[mnist_df['std_test_accuracy'] <= 1.0]
fashion_mnist_df = fashion_mnist_df[fashion_mnist_df['std_test_accuracy'] <= 1.0]
#%%
from matplotlib import pyplot as plt
# Make a plot with log num param as x-axis and test accuracy as y-axis.
# Include for the MPO (dataframe) data the SEM which is `std_test_accuracy / sqrt(num_experiments)`

# For each N value we plot the mean test accuracy with error bars using std.
# We also add the CPD data points as scatter points with lines connecting them.
plt.figure(figsize=(8, 6))
group_sorted = mnist_df.sort_values('num_parameters')
plt.errorbar(
    x=group_sorted['num_parameters'],
    y=group_sorted['avg_test_accuracy'],
    yerr=group_sorted['std_test_accuracy'],
    label=f'(MPO)$^2$',
    marker='o',
    linestyle='--'
)

# Add CPD data points
cpd_params = [row[2] for row in cpd_mnist]
cpd_acc = [row[3] for row in cpd_mnist]
plt.plot(cpd_params, cpd_acc, 'o--', label='CPD')

# Add TNML data points
tnml_params = [row[2] for row in tnml_mnist]
tnml_acc = [row[3] for row in tnml_mnist]
plt.plot(tnml_params, tnml_acc, 'o--', label='TNML')

plt.xscale('log')
plt.xlabel('Number of Parameters')
plt.ylabel('Test Accuracy (%)')
plt.title('MNIST: Test Accuracy vs Number of Parameters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=1.0)
plt.savefig('./figs/mnist_accuracy_vs_params.pdf', bbox_inches='tight', pad_inches=0)
# %%
# Fashion MNIST Plot
plt.figure(figsize=(8, 6))
group_sorted = fashion_mnist_df.sort_values('num_parameters')
plt.errorbar(
    x=group_sorted['num_parameters'],
    y=group_sorted['avg_test_accuracy'],
    yerr=group_sorted['std_test_accuracy'],
    label=f'(MPO)$^2$',
    marker='o',
    linestyle='--'
)

# Add CPD data points for Fashion MNIST
fashion_cpd_params = [row[2] for row in fashion_cpd]
fashion_cpd_acc = [row[3] for row in fashion_cpd]
plt.plot(fashion_cpd_params, fashion_cpd_acc, 'o--', label='CPD')

# Add TNML data points for Fashion MNIST
fashion_tnml_params = [row[2] for row in tnml_fashion]
fashion_tnml_acc = [row[3] for row in tnml_fashion]
plt.plot(fashion_tnml_params, fashion_tnml_acc, 'o--', label='TNML')

plt.xscale('log')
plt.xlabel('Number of Parameters')
plt.ylabel('Test Accuracy (%)')
plt.title('Fashion MNIST: Test Accuracy vs Number of Parameters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=1.0)
plt.savefig('./figs/fashion_mnist_accuracy_vs_params.pdf', bbox_inches='tight', pad_inches=0)

# %%
