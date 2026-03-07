import { describe, expect, inject, it } from 'vitest'
import { createAmazonBedrock } from '@ai-sdk/amazon-bedrock'
import { createOpenAI } from '@ai-sdk/openai'
import { createOpenResponses } from '@ai-sdk/open-responses'
import type { ToolSpec } from '@strands-agents/sdk'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '@strands-agents/sdk'
import type { ModelContentBlockStartEventData } from '$/sdk/models/streaming.js'
import { LanguageModel } from '$/sdk/models/language-model.js'
import { collectIterator } from '$/sdk/__fixtures__/model-test-helpers.js'

function createBedrockModel(
  modelId = 'us.anthropic.claude-sonnet-4-20250514-v1:0',
  config: { maxTokens?: number; temperature?: number } = {}
): LanguageModel {
  const credentials = inject('provider-language-model')?.credentials
  if (!credentials) throw new Error('No LanguageModel credentials provided')

  const provider = createAmazonBedrock({
    region: 'us-west-2',
    accessKeyId: credentials.accessKeyId,
    secretAccessKey: credentials.secretAccessKey,
    ...(credentials.sessionToken && { sessionToken: credentials.sessionToken }),
  })
  return new LanguageModel(provider(modelId), config)
}

function createMantleModel(
  modelId = 'openai.gpt-oss-120b',
  config: { maxTokens?: number; temperature?: number } = {}
): LanguageModel {
  const apiKey = inject('provider-mantle')?.apiKey
  if (!apiKey) throw new Error('No Bedrock Mantle API key provided')

  const provider = createOpenResponses({
    name: 'bedrock-mantle',
    url: 'https://bedrock-mantle.us-west-2.api.aws/v1/responses',
    apiKey,
  })
  return new LanguageModel(provider(modelId), config)
}

function createOpenAIModel(
  modelId = 'gpt-4o-mini',
  config: { maxTokens?: number; temperature?: number } = {}
): LanguageModel {
  const apiKey = inject('provider-openai')?.apiKey
  if (!apiKey) throw new Error('No OpenAI API key provided')

  const provider = createOpenAI({ apiKey })
  return new LanguageModel(provider(modelId), config)
}

interface SharedTestOptions {
  skipMaxTokens?: boolean
  skipUsage?: boolean
}

/**
 * Shared test suite for LanguageModel implementations.
 */
function languageModelTests(
  createModel: (modelId?: string, config?: { maxTokens?: number }) => LanguageModel,
  options: SharedTestOptions = {}
): void {
  describe('Basic text generation', () => {
    it.concurrent('generates a text response', async () => {
      const model = createModel(undefined, { maxTokens: 50 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Say hello.')] })]

      const events = await collectIterator(model.stream(messages))

      expect(events.find((e) => e.type === 'modelMessageStartEvent')).toBeDefined()
      expect(events.filter((e) => e.type === 'modelContentBlockDeltaEvent').length).toBeGreaterThan(0)
      expect(events.find((e) => e.type === 'modelMessageStopEvent')).toBeDefined()

      let text = ''
      for (const event of events) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
          text += event.delta.text
        }
      }
      expect(text.length).toBeGreaterThan(0)
    })

    it.concurrent('returns endTurn stop reason for natural completion', async () => {
      const model = createModel(undefined, { maxTokens: 100 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Say hi.')] })]

      const events = await collectIterator(model.stream(messages))
      const stopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
      expect(stopEvent?.stopReason).toBe('endTurn')
    })

    it.skipIf(options.skipMaxTokens).concurrent('returns maxTokens stop reason when limit reached', async () => {
      const model = createModel(undefined, { maxTokens: 16 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Write a long story about dragons.')] })]

      const events = await collectIterator(model.stream(messages))
      const stopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
      expect(stopEvent?.stopReason).toBe('maxTokens')
    })
  })

  describe('Usage and metadata', () => {
    it.skipIf(options.skipUsage).concurrent('reports token usage', async () => {
      const model = createModel(undefined, { maxTokens: 50 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Say hello.')] })]

      const events = await collectIterator(model.stream(messages))
      const metaEvent = events.find((e) => e.type === 'modelMetadataEvent')

      expect(metaEvent?.usage).toBeDefined()
      expect(metaEvent!.usage!.inputTokens).toBeGreaterThan(0)
      expect(metaEvent!.usage!.outputTokens).toBeGreaterThan(0)
      expect(metaEvent!.usage!.totalTokens).toBe(metaEvent!.usage!.inputTokens + metaEvent!.usage!.outputTokens)
    })
  })

  describe('Tool calling', () => {
    it.concurrent('returns toolUse stop reason when requesting tool use', async () => {
      const model = createModel(undefined, { maxTokens: 200 })

      const calculatorTool: ToolSpec = {
        name: 'calculator',
        description: 'Performs basic arithmetic. Use this to calculate math expressions.',
        inputSchema: {
          type: 'object',
          properties: { expression: { type: 'string', description: 'The math expression' } },
          required: ['expression'],
        },
      }

      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Use the calculator tool to compute 42 times 7. You must use the tool.')],
        }),
      ]
      const events = await collectIterator(model.stream(messages, { toolSpecs: [calculatorTool] }))

      const stopEvent = events.find((e) => e.type === 'modelMessageStopEvent')
      expect(stopEvent?.stopReason).toBe('toolUse')

      const toolStart = events.find((e) => e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart')
      expect(toolStart).toBeDefined()
    })

    it.concurrent('handles tool result round-trip', async () => {
      const model = createModel(undefined, { maxTokens: 200 })

      const calculatorTool: ToolSpec = {
        name: 'calculator',
        description: 'Performs basic arithmetic. Use this to calculate math expressions.',
        inputSchema: {
          type: 'object',
          properties: { expression: { type: 'string', description: 'The math expression' } },
          required: ['expression'],
        },
      }

      const messages1 = [new Message({ role: 'user', content: [new TextBlock('What is 6 * 7?')] })]
      const events1 = await collectIterator(model.stream(messages1, { toolSpecs: [calculatorTool] }))

      const toolStartEvent = events1.find(
        (e): e is ModelContentBlockStartEventData =>
          e.type === 'modelContentBlockStartEvent' && e.start?.type === 'toolUseStart'
      )
      const toolUseId = toolStartEvent?.start?.toolUseId ?? 'unknown'
      const toolName = toolStartEvent?.start?.name ?? 'calculator'

      let toolInput = ''
      for (const e of events1) {
        if (e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'toolUseInputDelta') {
          toolInput += e.delta.input
        }
      }

      const messages2 = [
        ...messages1,
        new Message({
          role: 'assistant',
          content: [
            new ToolUseBlock({ name: toolName, toolUseId, input: JSON.parse(toolInput || '{}') }),
            new ToolResultBlock({ toolUseId, status: 'success', content: [new TextBlock('42')] }),
          ],
        }),
      ]

      const events2 = await collectIterator(model.stream(messages2, { toolSpecs: [calculatorTool] }))

      let text = ''
      for (const e of events2) {
        if (e.type === 'modelContentBlockDeltaEvent' && e.delta.type === 'textDelta') {
          text += e.delta.text
        }
      }
      expect(text.toLowerCase()).toContain('42')
    })
  })

  describe('Content block lifecycle', () => {
    it.concurrent('emits start -> delta(s) -> stop in order', async () => {
      const model = createModel(undefined, { maxTokens: 50 })
      const messages = [new Message({ role: 'user', content: [new TextBlock('Say hello.')] })]

      const events = await collectIterator(model.stream(messages))

      const startIdx = events.findIndex((e) => e.type === 'modelContentBlockStartEvent')
      const firstDeltaIdx = events.findIndex((e) => e.type === 'modelContentBlockDeltaEvent')
      const stopIdx = events.findIndex((e) => e.type === 'modelContentBlockStopEvent')

      expect(startIdx).toBeLessThan(firstDeltaIdx)
      expect(firstDeltaIdx).toBeLessThan(stopIdx)
    })
  })

  describe('Error handling', () => {
    it.concurrent('throws on invalid model ID', async () => {
      const model = createModel('invalid-model-xyz-999')
      const messages = [new Message({ role: 'user', content: [new TextBlock('Hello')] })]

      await expect(collectIterator(model.stream(messages))).rejects.toThrow()
    })
  })
}

describe.skipIf(inject('provider-language-model').shouldSkip)('LanguageModel - Bedrock', () => {
  languageModelTests(createBedrockModel)
})

describe.skipIf(inject('provider-mantle').shouldSkip)('LanguageModel - Mantle', () => {
  languageModelTests(createMantleModel, { skipMaxTokens: true, skipUsage: true })
})

describe.skipIf(inject('provider-openai').shouldSkip)('LanguageModel - OpenAI', () => {
  languageModelTests(createOpenAIModel)
})
