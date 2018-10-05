library(qboxplot)
library(readxl)
Cleaned_Data_Characters <- read_excel("Cleaned_Data_Characters.xlsx")

test <- Cleaned_Data_Characters[2]

boxplot(test,
        horizontal=TRUE,las=1,
        main="Boxplot to Summarise the Number of Characters in Reviews After Preprocessing")

legend("top", inset=.05, title="Quartiles",
       c("Minimum = 3","1st Quartile = 29", "Median = 59","3rd Quartile = 124","Maximum = 5,740"), horiz=TRUE,
       cex=0.75)

