# How to Fine-tune Meta Llama-2-7B in Azure AI Studio

> [!TIP]
> See the [RAFT Distillation Recipe repository](https://github.com/0Upjh80d/raft-distillation-recipe) for instructions, notebooks and infrastructure provisioning for Meta Llama 3.1 and 3.2 as well as GPT-4o.

## Prerequisites

[Fine-tune Meta Llama models in Azure AI Foundry portal](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/fine-tune-model-llama?tabs=llama-three%2Cchatcompletion#prerequisites)

## Key Points

> [!IMPORTANT]
> Here are the key things to take note and get right in order for everything to work.

- Select the West US 3 location
- Use a Pay-As-You-Go (PAYG) subscription with billing set up
- Make sure the subscription is registered to the `Microsoft.Network` resource provider

## Getting Started (Step-by-Step Guide)

This builds on the [Fine-tune Meta Llama models in Azure AI Foundry portal](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/fine-tune-model-llama?tabs=llama-three%2Cchatcompletion#prerequisites) and adds a few details here and there.

1. Open [Azure AI Studio](https://ai.azure.com/).

2. Click on **$+$ New AI project** to create a new project.

   ![Step 01](images/azure-ai-studio-finetuning-01.png)

3. Enter a name under **Project name** > Click on **Create a new resource** to create a new AI Hub resource.

   ![Step 02](images/azure-ai-studio-finetuning-02.png)

4. Enter a name under **Azure AI Hub resource** > Under **Azure subscription**, select the subscription with PAYG billing > Select West US 3 for **Location**.

   ![Step 03](images/azure-ai-studio-finetuning-03.png)

> [!NOTE]
> It is important to use a PAYG subscription with billing set up. Grant based subscriptions and credits **will not** work.

5. Review and ensure that the information is correct and click on **Create an AI project**.

   ![Step 04](images/azure-ai-studio-finetuning-04.png)

6. The resources should begin creating.

   ![Step 05](images/azure-ai-studio-finetuning-05.png)

7. Wait until all the resources have been created.

   ![Step 06](images/azure-ai-studio-finetuning-06.png)

8. Once in the AI Studio project, select the **Fine-tuning** tab in the left panel > Click on **$+$ Fine-tune model**.

   ![Step 07](images/azure-ai-studio-finetuning-07.png)

9. Under **Select a model** > Select the model to fine-tune (e.g. Llama-2-7b) > Click **Confirm**.
   ![Step 08](images/azure-ai-studio-finetuning-08.png)

10. Click on **Subscribe and fine-tune** if necessary for the Meta subscription before starting the fine-tuning process.

    ![Step 09](images/azure-ai-studio-finetuning-09.png)

11. Under **Fine-tuned model name**, enter a name > Optionally, provide a description under **Description** > Click **Next**.

    ![Step 10](images/azure-ai-studio-finetuning-10.png)

12. Under the **Task type**, select a task type (currently, only text generation is supported).

    ![Step 11](images/azure-ai-studio-finetuning-11.png)

13. Under **Training data**, select the **Upload data** option > Upload your file (must be in `JSONL` format).

    ![Step 12](images/azure-ai-studio-finetuning-12.png)

14. Check **Overwrite if already exists**. The wizard will display the first few lines of the `JSONL` data file.
    ![Step 13](images/azure-ai-studio-finetuning-13.png)

15. Under **Required data columns**, select the appropriate columns for the prompt and the completion columns > Click **Next**.

    ![Step 14](images/azure-ai-studio-finetuning-14.png)

16. Select the appropriate task parameters > Click **Next**.

    ![Step 15](images/azure-ai-studio-finetuning-15.png)

17. Review the settings > Click **Submit** to submit the fine-tuning job.

    ![Step 16](images/azure-ai-studio-finetuning-16.png)

18. Under **Model attributes**, the job should have a **Running** status under **Status**.

    ![Step 17](images/azure-ai-studio-finetuning-17.png)

19. Wait until the job is completed.

    ![Step 18](images/azure-ai-studio-finetuning-18.png)
