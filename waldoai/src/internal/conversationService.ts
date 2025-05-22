// conversationService.ts
// Service for managing conversation history for Waldo AI

import { ChatMessage } from "@langchain/core/messages";

export interface ConversationTool {
  mtype: "tool",
  tool_id: string,
  text: string,
}

export interface ConversationMessage {
  mtype: "message",
  role: string;
  text: string;
  isThink: boolean;
}

export interface Conversation {
  messages: Array<ConversationTool | ConversationMessage>;
  mode: 'agent' | 'chat';
}

export type ConversationId = string;

export class ConversationService {

  private static instance: ConversationService;

  private constructor() {}

  public static getInstance(): ConversationService {
    if (!ConversationService.instance) {
      ConversationService.instance = new ConversationService();
    }
    return ConversationService.instance;
  }

  private conversations: Record<ConversationId, Conversation> = {
    "1747627580353": {
      messages: [
        { mtype: "message", role: "user", text: "How do I implement a React component?", isThink: false },
        { mtype: "tool", tool_id: "flash_think", text: "Using flash think to come up with an idea." },
        { mtype: "message", role: "assistant", text: "To implement a React component, you can use either a function or a class. Here's a simple functional component example...", isThink: false }
      ],
      mode: 'agent'
    },
    "1747627580354": {
      messages: [
        { mtype: "message", role: "user", text: "How do I create a REST API with Python?", isThink: false },
        { mtype: "message", role: "assistant", text: "You can create a REST API with Python using frameworks like Flask or FastAPI. Here's a simple example...", isThink: false }
      ],
      mode: 'agent'
    },
    "1747627580355": {
      messages: [
        { mtype: "message", role: "user", text: "How do I design a database schema for a blog?", isThink: false },
        { mtype: "message", role: "assistant", text: "For a blog database schema, you would typically need tables for users, posts, categories, comments, and tags. Here's a simple design...", isThink: false }
      ],
      mode: 'agent'
    }
  };

  getConversation(id: ConversationId): Conversation | undefined {
    return this.conversations[id];
  }

  setConversation(id: ConversationId, conversation: Conversation): void {
    this.conversations[id] = conversation;
  }

  deleteConversation(id: ConversationId): void {
    if (id !== 'current') {
      delete this.conversations[id];
    } else {
      this.conversations['current'] = { messages: [], mode: 'agent' };
    }
  }

  getAllConversations(): Record<ConversationId, Conversation> {
    return { ...this.conversations };
  }

  castMessagesToChatMessages(messages : ConversationMessage[]) : ChatMessage[] {
    return messages.map(msg => new ChatMessage(msg.text, msg.role));
  }

  castChatMessagesToMessages(messages : ChatMessage[]): ConversationMessage[] {
    return messages.map(msg => ({ mtype: "message", role: msg.role, text: msg.text, isThink: false }));
  }
}
