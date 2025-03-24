library(tidyverse)
library(patchwork)



accuracy = read.csv('output/accuracy.csv') %>%
  mutate(model_scenario = as.factor(model_scenario),
         aem_scenario = as.factor(aem_scenario),
         model = as.factor(model))


print(accuracy)

# Compute mean and range
summary_df <- accuracy %>%
  group_by(aem_scenario, model, model_scenario) %>%
  summarise(
    mean_accuracy = mean(accuracy),
    ymin = min(accuracy),  # Lower bound (can use mean - sd for confidence interval)
    ymax = max(accuracy)   # Upper bound (or mean + sd)
  ) %>% 
  mutate(aem_scenario = factor(aem_scenario, 
                               levels = c('PO4','NO3','NH4',
                                          'RSI', 'O2', 'CHLA', 'ZOOP','T',
                                          'T+NUTR', 'T+NUTR+MINOR',
                                          'T+NUTR+MINOR+SYSTEM')))

# Create the plot
ggplot(summary_df, aes(x = aem_scenario, y = mean_accuracy, color = model,  group = interaction(model, model_scenario))) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = model), alpha = 0.2, color = NA) +  # Shaded range
  geom_line(linewidth = 1) +  # Mean accuracy
  geom_point(size = 3) +  # Mean accuracy points
  geom_hline(yintercept = 0.3333, linetype = 'dashed') + ylim(0,1.0) +
  theme_minimal() +
  labs(title = "Mean Model Accuracy across AEM Scenarios",
       x = "",
       y = "Discriminator accuracy",
       color = "Model",
       linetype = "Model Scenario",
       fill = "Model") +  # Legend for shading
  theme(axis.text.x = element_text(angle = 20, hjust = 1), text = element_text(size = 15)) +
  facet_wrap(~ model_scenario, ncol = 1)
ggsave(filename = 'accuracy.png', dpi = 300, units = 'in', width = 14, height = 10)
