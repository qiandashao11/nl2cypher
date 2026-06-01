#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================== gui_app.py ====================
"""
Neo4j QA system - GUI version
Show the final answer and intermediate generated Cypher query
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from qa_system import Neo4jQASystem


class Neo4jQAApp:
    """Neo4j QA GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neo4j Knowledge Graph Q&A System")
        self.root.geometry("1000x700")
        
        # System state
        self.qa_system = None
        self.is_initialized = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the UI"""
        # ===== Configuration area =====
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # LoRA path
        ttk.Label(config_frame, text="LoRA Dir:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        self.lora_entry = ttk.Entry(config_frame, width=50)
        self.lora_entry.insert(0, "nl2cypher/non-parameter/lora_out_llama3_8b2")
        self.lora_entry.grid(row=0, column=1, padx=5, pady=3)
        
        # Neo4j configuration
        ttk.Label(config_frame, text="Neo4j URI:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        self.uri_entry = ttk.Entry(config_frame, width=50)
        self.uri_entry.insert(0, "neo4j://localhost:7687")
        self.uri_entry.grid(row=1, column=1, padx=5, pady=3)
        
        ttk.Label(config_frame, text="Database:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        self.db_entry = ttk.Entry(config_frame, width=50)
        self.db_entry.insert(0, "neo4j")
        self.db_entry.grid(row=2, column=1, padx=5, pady=3)
        
        ttk.Label(config_frame, text="Username:").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        self.user_entry = ttk.Entry(config_frame, width=20)
        self.user_entry.insert(0, "neo4j")
        self.user_entry.grid(row=3, column=1, sticky="w", padx=5, pady=3)
        
        ttk.Label(config_frame, text="Password:").grid(row=4, column=0, sticky="w", padx=5, pady=3)
        self.pass_entry = ttk.Entry(config_frame, width=20, show="*")
        self.pass_entry.insert(0, "neo4j")
        self.pass_entry.grid(row=4, column=1, sticky="w", padx=5, pady=3)
        
        # Language selection
        ttk.Label(config_frame, text="Language:").grid(row=5, column=0, sticky="w", padx=5, pady=3)
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(config_frame, textvariable=self.lang_var, 
                                   values=["English", "Chinese"], width=18, state="readonly")
        lang_combo.grid(row=5, column=1, sticky="w", padx=5, pady=3)
        
        # Initialize button
        self.init_btn = ttk.Button(config_frame, text="Initialize System", command=self._initialize_system)
        self.init_btn.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.status_label = ttk.Label(config_frame, text="Status: Not initialized", foreground="red")
        self.status_label.grid(row=7, column=0, columnspan=2)
        
        # ===== Question input area =====
        question_frame = ttk.LabelFrame(self.root, text="Question", padding=10)
        question_frame.pack(fill="x", padx=10, pady=5)
        
        self.question_entry = ttk.Entry(question_frame, width=80, font=("Arial", 11))
        self.question_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.question_entry.bind("<Return>", lambda e: self._ask_question())
        
        self.ask_btn = ttk.Button(question_frame, text="Ask", command=self._ask_question, state="disabled")
        self.ask_btn.pack(side="right", padx=5)
        
        # ===== Generated Cypher query display area =====
        cypher_frame = ttk.LabelFrame(self.root, text="Generated Cypher Query", padding=10)
        cypher_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.cypher_text = scrolledtext.ScrolledText(
            cypher_frame, 
            height=8, 
            font=("Courier", 10),
            bg="#f0f0f0",
            fg="#0000cc"
        )
        self.cypher_text.pack(fill="both", expand=True)
        
        # ===== Answer display area =====
        answer_frame = ttk.LabelFrame(self.root, text="Final Answer", padding=10)
        answer_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.answer_text = scrolledtext.ScrolledText(
            answer_frame, 
            height=12, 
            font=("Arial", 10),
            wrap="word"
        )
        self.answer_text.pack(fill="both", expand=True)
        
        # ===== Footer status bar =====
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.progress_label.pack(side="left")
    
    def _initialize_system(self):
        """Initialize QA system"""
        self.init_btn.config(state="disabled")
        self.status_label.config(text="Status: Initializing...", foreground="orange")
        self.progress_label.config(text="Initializing models...", foreground="orange")
        
        def init_thread():
            try:
                self.qa_system = Neo4jQASystem(
                    lora_dir=self.lora_entry.get(),
                    neo4j_uri=self.uri_entry.get(),
                    neo4j_user=self.user_entry.get(),
                    neo4j_password=self.pass_entry.get()
                )
                self.is_initialized = True
                
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Ready ✓", foreground="green"
                ))
                self.root.after(0, lambda: self.progress_label.config(
                    text="System ready", foreground="green"
                ))
                self.root.after(0, lambda: self.ask_btn.config(state="normal"))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", "System initialized successfully!"
                ))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text="Status: Initialization failed", foreground="red"
                ))
                self.root.after(0, lambda: self.progress_label.config(
                    text="Error", foreground="red"
                ))
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to initialize system:\n{str(e)}"
                ))
            finally:
                self.root.after(0, lambda: self.init_btn.config(state="normal"))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def _ask_question(self):
        """Handle the question"""
        if not self.is_initialized:
            messagebox.showwarning("Warning", "Please initialize the system first!")
            return
        
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Warning", "Please enter a question!")
            return
        
        # Clear previous results
        self.cypher_text.delete(1.0, tk.END)
        self.answer_text.delete(1.0, tk.END)
        
        # Disable buttons
        self.ask_btn.config(state="disabled")
        self.progress_label.config(text="Processing...", foreground="orange")
        
        def qa_thread():
            try:
                # Run QA
                result = self.qa_system.answer(
                    question=question,
                    database=self.db_entry.get(),
                    language=self.lang_var.get(),
                    verbose=False
                )
                
                # Show Cypher
                self.root.after(0, lambda: self.cypher_text.insert(1.0, result['cypher']))
                
                # Show answer
                self.root.after(0, lambda: self.answer_text.insert(1.0, result['answer']))
                
                # Update status
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"Query returned {result['results']['count']} records", 
                    foreground="green"
                ))
                
            except Exception as e:
                self.root.after(0, lambda: self.answer_text.insert(
                    1.0, f"Error: {str(e)}"
                ))
                self.root.after(0, lambda: self.progress_label.config(
                    text="Error occurred", foreground="red"
                ))
            finally:
                self.root.after(0, lambda: self.ask_btn.config(state="normal"))
        
        threading.Thread(target=qa_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = Neo4jQAApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()