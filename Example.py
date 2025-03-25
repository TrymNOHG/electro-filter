Len = 9


test1 = np.arange(Len**2 - Len) + 1 + 0
test2 = np.arange(Len**2 - Len) + 1 + 3
testdiag = np.arange(Len**2) + 1
testdiag1 = np.arange(Len**2 - 1) + 1 + 0
testdiag2 = np.arange(Len**2 - 1) + 1 + 1

kek = np.zeros((Len**2, Len**2))
print(kek)
print(test)
kek = kek + np.diag(testdiag)
kek = kek + np.diag(testdiag1, 1)
kek = kek + np.diag(testdiag2, -1)
kek = kek + np.diag(test1, Len)
kek = kek + np.diag(test2, -Len)
print(kek)

for i in range(Len-1):
    kek[(i+1)*Len][(i+1)*Len-1] = 0
    kek[(i+1)*Len-1][(i+1)*Len] = 0
print(kek)

kek = np.concatenate(kek)


direction = np.arange(Len**2) + 1
ones = np.ones(Len**2)

dir1 = np.reshape(ones, (Len**2, 1))
dir2 = np.reshape(direction, (1, Len**2))

xdir = np.dot(dir1, dir2)
xdir = np.concatenate(xdir)

ydir = np.dot(np.transpose(dir2), np.transpose(dir1))
ydir = np.concatenate(ydir)

print(xdir)
print(ydir)




# Create a scatter plot with color variation
plt.scatter(xdir, ydir, c=kek, cmap='viridis', s=100)  # 'c' specifies the color, 's' is the size of the dots
plt.colorbar()  # To show the color scale
plt.xlabel('X values')
plt.ylabel('Y values')
plt.gca().invert_yaxis()
plt.title('Scatter Plot with Colored Dots')
plt.show()
