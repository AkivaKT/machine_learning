---
title: "Prove_8(Neural Network)"
author: "Keith Tung"
date: "March 09, 2019"
output:
  html_document:  
    keep_md: true
    toc: true
    toc_float: true
    code_folding: hide
    fig_height: 6
    fig_width: 12
    fig_align: 'center'
---






```r
# Use this R-Chunk to import all your datasets!
output <- read_csv("output_1.csv", 
    col_names = paste0("V",seq_len(500)))

output_2 <- read_csv("output_2.csv", 
    col_names = paste0("V",seq_len(500)))

output_ <- read_csv("output__1.csv", 
    col_names = paste0("V",seq_len(500)))

output__2 <- read_csv("output__2.csv", 
    col_names = paste0("V",seq_len(500)))
```

## Data Wrangling


```r
# Use this R-Chunk to clean & wrangle your data!
output <- output %>% 
  mutate(loop = row_number()) %>%
  select(loop, 1:500) %>% 
  gather(key = 'times', value = 'precent', c(2:501)) %>% 
  mutate(times = as.numeric(str_remove_all(.$times, 'V')))


output_2 <- output_2 %>% 
  mutate(loop = row_number()) %>%
  select(loop, 1:500) %>% 
  gather(key = 'times', value = 'precent', c(2:501)) %>% 
  mutate(times = as.numeric(str_remove_all(.$times, 'V'))) 

output_ <- output_ %>% 
  mutate(loop = row_number()) %>%
  select(loop, 1:500) %>% 
  gather(key = 'times', value = 'precent', c(2:501)) %>% 
  mutate(times = as.numeric(str_remove_all(.$times, 'V'))) 


output__2 <- output__2 %>% 
  mutate(loop = row_number()) %>%
  select(loop, 1:500) %>% 
  gather(key = 'times', value = 'precent', c(2:501)) %>% 
  mutate(times = as.numeric(str_remove_all(.$times, 'V'))) 
```



## Source code

Provide a link to your GitHub repository for the code you wrote.  
  
https://github.com/AkivaKT/machine_learning/tree/master/Prove_7  
## Approach

I picked the implementation approach and my classifier will learn and take in a validation set to compare. I utilized objects for my algorithm and store the activation value and error. My classifier will build a network according to the input and output while the user could decide how many layers and nodes should the network builds. 
    Challenges
The toughest part, in my opinion, is the building part, while the learning/ training part involves some calculation and data manipulation, the building part takes into account many aspects. It took me awhile tell the program to build proper numbers of weights. I also spent some time to understand how a dot product works with two vectors.


## Results


Make sure to include evidence (description, graphs, results, etc.) that demonstrate each of the items listed in the minimum standards above.

iris data:  
two hidden layers with 4 nodes each  
  
I looped through the training data 8 times with different starting weights to avoid reaching the local min of error. In general, the model was learning pretty fast, there were 3 models that reach 100% scoring from the validation set. There were two networks that got stuck for a while until it started learning.  
  
the score of the test data: 89%  

```r
output %>%   
filter(!is.na(precent)) %>% 
ggplot() +
  aes(y = as.numeric(precent) * 100, x = times, group = as.factor(loop), colour = as.factor(loop)) +
  geom_smooth(se = F, size = 1.15) +
  scale_color_brewer(palette = 'Dark2') +
  ggthemes::theme_fivethirtyeight()
```

![](present_files/figure-html/plot_data-1.png)<!-- -->
iris data:  
one hidden layer with 5 nodes.   
  
this setting helps the model learn even faster, all loop reached its max within 100 loops. This setting gives a better score at 96.3%.  

```r
output_2 %>% 
  ggplot() +
  aes(y = as.numeric(precent) * 100, x = times, group = as.factor(loop), colour = as.factor(loop)) +
  geom_smooth(se = F, size = 1.15) +
  scale_color_brewer(palette = 'Accent') +
  ggthemes::theme_fivethirtyeight()
```

![](present_files/figure-html/unnamed-chunk-2-1.png)<!-- -->
  
Predicting political party,  
two layers with 5 nodes each.  
Unlike the last dataset, two out of the five times, this model showed a fast learning rate than the one layer setting. However, the other three times, the network was stuck and never improved, two of which was stopped by the classifier since it was showing no progress.  
  
score: 92.6%  


```r
output_ %>%   
filter(!is.na(precent)) %>% 
ggplot() +
  aes(y = as.numeric(precent) * 100, x = times, group = as.factor(loop), colour = as.factor(loop)) +
  geom_smooth(se = F, size = 1.15) +
  scale_color_brewer(palette = 'Dark2') +
  ggthemes::theme_fivethirtyeight()
```

![](present_files/figure-html/unnamed-chunk-3-1.png)<!-- -->
  
one layer with 10 nodes.  
  
These one layer networks show a better curve in improving, although taking longer time, three out of five of the attempts had a decent score.  
  
score:100%  

```r
output__2 %>% 
  ggplot() +
  aes(y = as.numeric(precent) * 100, x = times, group = as.factor(loop), colour = as.factor(loop)) +
  geom_smooth(se = F, size = 1.15) +
  scale_color_brewer(palette = 'Accent') +
  ggthemes::theme_fivethirtyeight()
```

![](present_files/figure-html/unnamed-chunk-4-1.png)<!-- -->


## Above and Beyond

Describe any efforts you made to go above and beyond.  
I set up different stopping criteria for my classifier, if the valid set's accuracy is dropping too much or too fast, it will stop and keep the best-performing weights. However, I do realize that sometimes models take a longer time to train; I then set up an allowance of 250 loops without improving the valid set's score and when it stops improving after 250 loops, it will stop again. That's why you can see how some model stopped right after a major drop of scoring in the graphs. 



E) Shows creativity and excels above and beyond requirements. (I am late for a week.)  



