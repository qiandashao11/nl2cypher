
import argparse
from qa_system import Neo4jQASystem
from rich.console import Console
from rich.panel import Panel


console = Console()


def print_header():
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║    Neo4j Knowledge Graph Q&A System                  ║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]\n")


def interactive_mode(qa_system, database):
    print_header()
    console.print("[green]✓ System initialized successfully![/green]")
    console.print("[yellow]Type 'exit' or 'quit' to exit, 'clear' to clear screen[/yellow]\n")
    
    while True:
        try:
            console.print("[bold blue]Your Question:[/bold blue] ", end="")
            question = input().strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            if question.lower() == 'clear':
                console.clear()
                print_header()
                continue
            
            console.print("\n[cyan]Processing...[/cyan]")
            result = qa_system.answer(
                question=question,
                database=database,
                verbose=False
            )
            
            console.print("\n" + "="*60)
            console.print(Panel(
                result['cypher'],
                title="[bold green]Generated Cypher Query[/bold green]",
                border_style="green"
            ))
            
            if result['results']['success']:
                console.print(f"[dim]→ Query returned {result['results']['count']} records[/dim]")
            else:
                console.print(f"[red]→ Query failed: {result['results'].get('error')}[/red]")
            
            console.print("\n" + "="*60)
            console.print(Panel(
                result['answer'],
                title="[bold cyan]Final Answer[/bold cyan]",
                border_style="cyan"
            ))
            console.print("="*60 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]\n")
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]\n")


def single_query_mode(qa_system, question, database):
    print_header()
    
    console.print(f"[bold blue]Question:[/bold blue] {question}\n")
    console.print("[cyan]Processing...[/cyan]\n")
    
    result = qa_system.answer(
        question=question,
        database=database,
        verbose=True
    )
    
    console.print("\n" + "="*60)
    console.print(Panel(
        result['cypher'],
        title="[bold green]Generated Cypher Query[/bold green]",
        border_style="green"
    ))
    
    if result['results']['success']:
        console.print(f"[dim]→ Query returned {result['results']['count']} records[/dim]")
    else:
        console.print(f"[red]→ Query failed: {result['results'].get('error')}[/red]")
    
    console.print("\n" + "="*60)
    console.print(Panel(
        result['answer'],
        title="[bold cyan]Final Answer[/bold cyan]",
        border_style="cyan"
    ))
    console.print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j Knowledge Graph Q&A System - CLI"
    )
    parser.add_argument("--question", "-q", help="Single question to ask")
    parser.add_argument("--lora_dir", default="lora_out_llama3_8b4")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--neo4j_uri", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j_user", default="neo4j")
    parser.add_argument("--neo4j_password", default="neo4j")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode")
    args = parser.parse_args()
    
    console.print("[cyan]Initializing system...[/cyan]")
    qa_system = Neo4jQASystem(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    if args.question:
        single_query_mode(qa_system, args.question, args.database)
    else:
        interactive_mode(qa_system, args.database)
    
    qa_system.close()


if __name__ == "__main__":
    main()
