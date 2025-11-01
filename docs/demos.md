# Live Demos

Explore Portico's capabilities through interactive demo applications. Each demo showcases specific features and architectural patterns, allowing you to experience how Portico handles real-world application scenarios.

All demos are fully deployed and running on Cloud Run with production-grade infrastructure.

## RBAC Demo: Organization Webapp

**Purpose:** Demonstrates hierarchical role-based access control (RBAC) in a corporate structure using a fictional company called "TechCorp."

### What It Demonstrates

This application shows how Portico handles complex permission hierarchies where access rights flow through organizational structures. Users can create groups (divisions, teams, projects), assign roles with specific permissions, and see how permission inheritance works across the hierarchy.

### Portico Kits Used

- **User Kit** - User account management and authentication
- **Auth Kit** - Session-based authentication with secure cookies
- **Group Kit** - Hierarchical group/organization management
- **RBAC Kit** - Role-based access control with permission inheritance
- **Audit Kit** - Comprehensive logging of all user actions and permission checks
- **File Storage Kit** - File uploads with group-based access control

### Key Features

- **Hierarchical Groups** - Company → Division → Team → Project structure
- **Permission Inheritance** - Permissions flow down through the hierarchy
- **Custom Roles** - Define roles with specific permission sets
- **Audit Trail** - Every action is logged with full context
- **File Management** - Upload and manage files with access control
- **User Administration** - Create users and assign them to groups with specific roles

### Architecture Highlights

The organization webapp demonstrates the repository pattern with complete separation between business logic (kits) and database operations (adapters). All permission checks are handled through the RBAC service, which queries the group hierarchy to determine access rights.

### Try It Live

**Demo URL:** [https://rbac.demos.portico.build](https://rbac.demos.portico.build)

**Demo Credentials:**
- Email: `demo@techcorp.com`
- Password: `Demopassword123!`

**What to Try:**
- Navigate the organizational hierarchy (Company → Division → Team)
- Create a new team or project
- Upload files and see access control in action
- Check the audit log to see all tracked actions
- Try accessing resources in different groups to see permission inheritance

---

## Cache Demo: Twitter Clone

**Purpose:** Demonstrates intelligent caching strategies and cache invalidation patterns in a social media application.

### What It Demonstrates

This Twitter-like application shows how Portico's caching system can dramatically improve performance while maintaining data consistency. It implements smart cache invalidation using tags, ensuring users always see fresh content when relationships change or new posts are created.

### Portico Kits Used

- **User Kit** - User profiles and account management
- **Auth Kit** - Session-based authentication
- **Cache Kit** - Redis-backed caching with tag-based invalidation
- **Audit Kit** - Activity logging for tweets, follows, and profile updates
- **Database Adapter** - SQLAlchemy with async support for PostgreSQL/SQLite

### Key Features

- **Smart Feed Caching** - User timelines cached for 5 minutes, invalidated on new tweets
- **Profile Caching** - User profiles cached with automatic invalidation on updates
- **Tag-Based Invalidation** - When you follow someone, relevant caches are cleared instantly
- **Cache Metrics Dashboard** - Real-time view of cache hit/miss ratios and performance
- **Follow Relationships** - Social graph with optimized queries through caching
- **Audit Logging** - Complete trail of all user actions

### Architecture Highlights

The Twitter demo showcases cache-aside pattern with tag-based invalidation. When a user posts a tweet, caches tagged with that user's ID are automatically invalidated. The application demonstrates how to balance freshness requirements with performance gains, achieving ~70% cache hit ratios in realistic scenarios.

### Try It Live

**Demo URL:** [https://cache.demos.portico.build](https://cache.demos.portico.build)

**Demo Credentials:** (password: `Password123!` for all)

- `alice@example.com` - Platform administrator
- `bob@example.com` - Software developer
- `charlie@example.com` - Travel blogger
- `diana@example.com` - Food critic
- `eve@example.com` - Tech entrepreneur
- `frank@example.com` - Musician

**What to Try:**
- Post tweets and see real-time feed updates
- Follow/unfollow users and observe cache invalidation
- Check `/cache/metrics` to see cache performance statistics
- View profiles to see cached data in action
- Compare public timeline vs personalized feed caching strategies

---

## LLM Lab: Template & RAG Demo

**Purpose:** Demonstrates template management with automatic versioning and LLM-powered conversations with document-based retrieval-augmented generation (RAG).

### What It Demonstrates

This application shows how Portico handles versioned prompt templates, variable substitution with Jinja2, and multi-modal LLM conversations. It integrates with Graphlit for managed RAG, allowing conversations to reference uploaded documents through semantic search.

### Portico Kits Used

- **Template Kit** - Template creation, editing, and automatic versioning
- **LLM Kit** - OpenAI integration for chat completions
- **Conversation Service** - Message history and context management
- **Audit Kit** - Action logging for templates and conversations
- **File Storage Kit** - Google Cloud Storage for document uploads
- **Managed RAG** - Graphlit integration for document processing and retrieval

### Key Features

- **Template Versioning** - Every template change creates a new version automatically
- **Variable Extraction** - Jinja2 variables (`{{variable_name}}`) automatically detected
- **Template Rollback** - Restore any previous template version with one click
- **Template-Based Conversations** - Create LLM chats from templates with variable substitution
- **Document Modes** - Three conversation types:
  - No Documents - Standard chat
  - Single Doc Spotlight - Deep dive on specific document
  - Knowledge Base Search - RAG across multiple documents
- **Conversation History** - Full message tracking with template context preserved

### Architecture Highlights

The LLM Lab demonstrates repository pattern without a dedicated kit layer. Templates use the `SqlAlchemyTemplateRepository` directly, showing how Portico supports flexible architectural patterns. The automatic versioning feature is implemented at the repository level, creating version snapshots on every update without requiring service-layer code.

### Try It Live

**Demo URL:** [https://lab.demos.portico.build](https://lab.demos.portico.build)

**What to Try:**
- Create a template with Jinja2 variables: `Hello {{name}}, welcome to {{company}}!`
- Edit the template and observe automatic version creation
- View version history and restore an old version
- Start a conversation from a template by filling in variable values
- Upload documents and try Knowledge Base Search mode
- Check how template context is preserved even if original template changes

---

## TaskFlow: Project Management Demo

**Purpose:** Demonstrates building feature-rich task management applications with Portico's core capabilities.

### What It Demonstrates

This Kanban-style project management application shows how to build traditional CRUD applications with Portico. It includes task tracking, priority management, workflow status transitions, and project organization. Future phases will add background jobs for notifications and AI-powered blocker detection.

### Portico Kits Used

- **User Kit** - User account management
- **Auth Kit** - Session-based authentication
- **Database Adapter** - SQLAlchemy with async PostgreSQL support
- **FastAPI Integration** - Exception handlers, static files, template rendering

**Planned for Future Phases:**

- **Job Kit** - Background job processing for notifications (Phase 3)
- **LLM Kit** - AI-powered blocker detection and suggestions (Phase 4)

### Key Features

- **Project Management** - Create and organize multiple projects
- **Kanban Board** - Visual workflow with drag-and-drop (To Do → In Progress → Blocked → Done)
- **Priority Levels** - Five priority levels with visual indicators
- **Task Lifecycle** - Automatic timestamp tracking (created_at, started_at, completed_at)
- **Dashboard** - Project statistics and quick actions
- **Task Status Workflow** - Manage task state transitions with validation

### Architecture Highlights

TaskFlow demonstrates a clean separation between domain models and database models. Custom `ProjectModel` and `TaskModel` classes handle the application-specific data, while Portico's core tables manage users and sessions. The application follows repository pattern with explicit session management.

### Try It Live

**Demo URL:** [https://tasks.demos.portico.build](https://tasks.demos.portico.build)

**Demo Credentials:**
- Email: `demo@taskflow.local`
- Password: `Demo123!`

**What to Try:**
- Create a new project and add tasks
- Move tasks through the Kanban workflow
- Set different priority levels and observe visual indicators
- View the dashboard to see project statistics
- Complete tasks and see automatic timestamp updates
- Create tasks with dependencies and relationships (coming in Phase 2)

---

## Running Demos Locally

All demo applications are available in the [portico-examples](https://github.com/PorticoFoundry/portico-examples) repository. Each demo can be run locally for development and testing.

### Quick Start

```bash
# Clone the examples repository
git clone https://github.com/PorticoFoundry/portico-examples.git
cd portico-examples

# Initialize submodules (each example has portico as a submodule)
make init-submodules

# Run a specific demo
cd examples/twitter_webapp
poetry install
poetry run python seed_data.py  # Load demo data
poetry run python main.py       # Start the app

# Open browser to http://localhost:8003
```

### Available Demos

- **Organization Webapp** - Port 8004, SQLite database
- **Twitter Clone** - Port 8003, SQLite + Redis
- **LLM Lab** - Port 8007, PostgreSQL + GCS + Graphlit
- **TaskFlow** - Port 8007, PostgreSQL + APScheduler

Each demo includes:

- Complete source code
- Database seeding scripts
- Integration tests
- Deployment configuration (Dockerfile, Cloud Run setup)
- Development documentation in `CLAUDE.md`

---

## Demo Infrastructure

All demos are deployed on Google Cloud Run with production-grade infrastructure:

- **Auto-scaling** - Scales from 0 to N instances based on traffic
- **HTTPS/TLS** - Automatic SSL certificate provisioning and renewal
- **PostgreSQL** - Cloud SQL with private VPC networking
- **Redis** - Memorystore for cache backend
- **File Storage** - Google Cloud Storage for uploads
- **Monitoring** - Cloud Logging and Error Reporting
- **CI/CD** - Automated builds via Cloud Build

The infrastructure is defined as code using Terraform, available in the `portico-examples/terraform/` directory.

---

## What You'll Learn

By exploring these demos, you'll see how Portico:

### Enforces Clean Architecture
- Business logic (kits) never imports infrastructure (adapters)
- Composition root pattern keeps dependencies explicit
- Repository pattern separates domain models from database models

### Handles Real-World Scenarios
- Hierarchical permissions with inheritance (RBAC Demo)
- Cache invalidation strategies (Twitter Demo)
- Template versioning and rollback (LLM Lab)
- CRUD operations with workflow states (TaskFlow)

### Integrates with External Services
- OpenAI for LLM completions
- Graphlit for managed RAG
- Redis for distributed caching
- PostgreSQL for relational data
- Google Cloud Storage for file uploads

### Maintains Code Quality
- Type hints throughout
- Comprehensive test coverage
- Structured logging with context
- Exception handling patterns
- Audit logging for compliance

---

## Contributing Demo Improvements

Found a bug or want to suggest an enhancement? Each demo is maintained in the [portico-examples](https://github.com/PorticoFoundry/portico-examples) repository.

### Reporting Issues

1. Check if the issue already exists
2. Create a new issue with:

   - Demo name (Organization, Twitter, LLM Lab, or TaskFlow)
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable

### Suggesting Features

We welcome suggestions for:

- New demo applications showcasing different Portico features
- Improvements to existing demos
- Additional use cases or scenarios

Open an issue in the portico-examples repository with the `enhancement` label.

---

## Next Steps

After exploring the demos:

1. **Read the Architecture Guide** - Understand [Portico's philosophy](philosophy.md) and hexagonal architecture
2. **Explore Ports** - Learn about the [available ports](ports/index.md) and their interfaces
3. **Study Kits** - See how [kits](kits/index.md) compose ports into features
4. **Build Your Own** - Start with the [Portico documentation](index.md) to create your first application

The demos are fully open-source and serve as reference implementations. Feel free to use them as starting points for your own applications.
