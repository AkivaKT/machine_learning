library(tidyverse)
library(ggplot2)
library(stringr)
output %>% 
  mutate(loop = row_number()) %>%
  select(loop, 1:251) %>% 
  gather(key = 'times', value = 'precent', c(2:252)) %>% 
  mutate(times = as.numeric(str_remove_all(.$times, 'X'))) %>% 
  ggplot() +
  aes(y = precent * 100, x = times, group = as.factor(loop), colour = as.factor(loop)) +
  geom_smooth(se = F) +
  scale_color_brewer(palette = 'Dark2') +
  ggthemes::theme_fivethirtyeight()

ggsave('plot1.png')


output_best %>% 
  mutate(times = row_number(),
         precent = X1) %>% 
  ggplot()+
  aes(y = precent * 100, x = times) +
  geom_smooth(se = F) +
  scale_color_brewer(palette = 'Dark2') +
  ggthemes::theme_fivethirtyeight()
ggsave('plot1_best.png')





x <- read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
         col_names = F, na = '?') 

x1 <- x %>% 
  filter(X1 == "republican") %>% 
  mutate(X2 = case_when(is.na(X2)   ~  max(X2, na.rm = T),
                        T           ~  X2),
         X3 = case_when(is.na(X3)   ~  max(X3, na.rm = T),
                        T           ~  X3),
         X4 = case_when(is.na(X4)   ~  max(X4, na.rm = T),
                        T           ~  X4),
         X5 = case_when(is.na(X5)   ~  max(X5, na.rm = T),
                        T           ~  X5),
         X6 = case_when(is.na(X6)   ~  max(X6, na.rm = T),
                        T           ~  X6),
         X7 = case_when(is.na(X7)   ~  max(X7, na.rm = T),
                        T           ~  X7),
         X8 = case_when(is.na(X8)   ~  max(X8, na.rm = T),
                        T           ~  X8),
         X9 = case_when(is.na(X9)   ~  max(X9, na.rm = T),
                        T           ~  X9),
         X10 = case_when(is.na(X10) ~  max(X10, na.rm = T),
                        T           ~  X10),
         X11 = case_when(is.na(X11)  ~  max(X11, na.rm = T),
                        T           ~  X11),
         X12 = case_when(is.na(X12)  ~  max(X12, na.rm = T),
                        T           ~  X12),
         X13 = case_when(is.na(X13)  ~  max(X13, na.rm = T),
                        T           ~  X13),
         X14 = case_when(is.na(X14)  ~  max(X14, na.rm = T),
                        T           ~  X14),
         X15 = case_when(is.na(X15)  ~  max(X15, na.rm = T),
                        T           ~  X15),
         X16 = case_when(is.na(X16)  ~  max(X16, na.rm = T),
                        T           ~  X16),
         X17 = case_when(is.na(X17)  ~  max(X17, na.rm = T),
                        T           ~  X17))

x1[x1 == 'n'] <- 0



 x2 <- x %>% 
  filter(X1 == "democrat") %>% 
  mutate(X2 = case_when(is.na(X2)   ~  max(X2, na.rm = T),
                        T           ~  X2),
         X3 = case_when(is.na(X3)   ~  max(X3, na.rm = T),
                        T           ~  X3),
         X4 = case_when(is.na(X4)   ~  max(X4, na.rm = T),
                        T           ~  X4),
         X5 = case_when(is.na(X5)   ~  max(X5, na.rm = T),
                        T           ~  X5),
         X6 = case_when(is.na(X6)   ~  max(X6, na.rm = T),
                        T           ~  X6),
         X7 = case_when(is.na(X7)   ~  max(X7, na.rm = T),
                        T           ~  X7),
         X8 = case_when(is.na(X8)   ~  max(X8, na.rm = T),
                        T           ~  X8),
         X9 = case_when(is.na(X9)   ~  max(X9, na.rm = T),
                        T           ~  X9),
         X10 = case_when(is.na(X10) ~  max(X10, na.rm = T),
                         T           ~  X10),
         X11 = case_when(is.na(X11)  ~  max(X11, na.rm = T),
                         T           ~  X11),
         X12 = case_when(is.na(X12)  ~  max(X12, na.rm = T),
                         T           ~  X12),
         X13 = case_when(is.na(X13)  ~  max(X13, na.rm = T),
                         T           ~  X13),
         X14 = case_when(is.na(X14)  ~  max(X14, na.rm = T),
                         T           ~  X14),
         X15 = case_when(is.na(X15)  ~  max(X15, na.rm = T),
                         T           ~  X15),
         X16 = case_when(is.na(X16)  ~  max(X16, na.rm = T),
                         T           ~  X16),
         X17 = case_when(is.na(X17)  ~  max(X17, na.rm = T),
                         T           ~  X17))




x3 <- bind_rows(x1,x2) %>% as.data.frame()


x3[x3 == 'n'] <- 0

write_csv(x3, path = 'machine_learning/Prove_7/data_2.csv')
output %>% 
  filter(!is.na(precent)) %>% 
  group_by(loop) %>% 
  arrange(loop, times) %>% 
  summarise(times = max(times),
            times)


