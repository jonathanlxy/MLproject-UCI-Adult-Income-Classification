require(data.table)
require(caTools)
# ==== Read file ============================================
Adult.data <- fread('adult.data.txt')
adult <- as.data.frame(Adult.data)
# Benchmark using majority vote
# 0.7591904
# max(table(Adult.data$V15))/nrow(Adult.data)

# ==== Data preparation =====================================
# Transform char into int
adultcol <- colnames(adult)
charIndex <- sapply(adult, class) == 'character'
char_int <- as.data.frame(sapply(names(which(charIndex == T)), # Colnames of factor columns
                                 function(i) as.integer(as.factor(adult[, i])) - 1 )
                          )
IntSet <- cbind(adult[, charIndex == F], char_int)

# Re-order
IntSet <- IntSet[, adultcol]

# Remove column(s)
IntSet$V4 <- NULL    # remove education

# Scaling
scaSet <- scale(IntSet[, 1:(ncol(IntSet)-1)])
y <- IntSet[, ncol(IntSet)] # To prevent duplicated "V14" issue

# Re-combine
IntSet <- as.data.frame(cbind(scaSet,y))
FinalSet <- IntSet

# === Split ====
set.seed(42)
spt <- sample.split(FinalSet$y, .75)
train <- subset(FinalSet, spt == T)
test  <- subset(FinalSet, spt == F)

# Train glm model using glm
LogMod <- glm(y ~ ., data = train, family = 'binomial')
pred <- predict(LogMod, newdata = test[,1:13], type = 'response')
pred[pred>.5] = 1
pred[pred<.5] = 0
max(table(pred == test$y))/nrow(test)


write.csv(train, file = "finalset_cleaned_train.csv", row.names = F)
write.csv(test,  file = "finalset_cleaned_test.csv", row.names = F)