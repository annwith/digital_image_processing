# Commit Convention

We follow the conventional commits specification for our project's commit messages. This helps us maintain a standardized and readable format across all commits.

## Format
Each commit message should follow this format:

<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

### Types
- **feat**: A new feature or enhancement.
- **fix**: A bug fix.
- **docs**: Documentation changes.
- **style**: Changes that do not affect the meaning of the code (e.g., formatting, white-space changes, etc.).
- **refactor**: Code changes that neither fix a bug nor add a feature.
- **test**: Adding missing tests or correcting existing tests.
- **chore**: Changes to the build process, auxiliary tools, or libraries such as documentation generation.

### Examples
- `feat(auth): Add user authentication feature`
- `fix(ui): Fix alignment issue in sidebar`
- `docs(readme): Update installation instructions`
- `style(css): Format stylesheets according to coding standards`
- `refactor(api): Improve error handling in API`
- `test(api): Add unit tests for API endpoints`
- `chore(deps): Update dependency versions`

## Further Reading
For more information on conventional commits, refer to the [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/).
