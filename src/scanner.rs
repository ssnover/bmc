use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind<'a> {
    Ident(&'a str),
    Keyword(Keyword),
    Colon,
    Whitespace,
    AssignmentOp,
    Semicolon,
    IntegerLiteral(&'a str),
    CharLiteral(&'a str),
    StringLiteral(&'a str),
    LogicalAnd,
    LogicalOr,
    PostIncrement,
    PostDecrement,
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
    //UnaryNegation, // not sure about this one
    LogicalNot,
    Exponentiation,
    Multiply,
    Divide,
    Modulo,
    Add,
    Subtract,
    LessThan,
    GreaterThan,
    LeftCurlyBrace,
    RightCurlyBrace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LineComment(&'a str),
    BlockComment(&'a str),
    Invalid(Invalid),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Keyword {
    Array,
    Boolean,
    Char,
    Else,
    False,
    For,
    Function,
    If,
    Integer,
    Print,
    Return,
    String,
    True,
    Void,
    While,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Invalid {
    CharLiteralTooLong,
    EmptyCharLiteral,
    UnexpectedChar,
    UnterminatedBlockComment,
    UnterminatedCharLiteral,
    UnterminatedStringLiteral,
}

impl FromStr for Keyword {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "array" => Keyword::Array,
            "boolean" => Keyword::Boolean,
            "char" => Keyword::Char,
            "else" => Keyword::Else,
            "false" => Keyword::False,
            "for" => Keyword::For,
            "function" => Keyword::Function,
            "if" => Keyword::If,
            "integer" => Keyword::Integer,
            "print" => Keyword::Print,
            "return" => Keyword::Return,
            "string" => Keyword::String,
            "true" => Keyword::True,
            "void" => Keyword::Void,
            "while" => Keyword::While,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq)]
pub struct SpanBound {
    line: usize,
    col: usize,
}

impl SpanBound {
    pub fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

impl From<(usize, usize)> for SpanBound {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl Ord for SpanBound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.line.cmp(&other.line) {
            std::cmp::Ordering::Equal => self.col.cmp(&other.col),
            ord => ord,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    start: SpanBound,
    end: SpanBound,
}

impl Span {
    pub fn new(start: SpanBound, end: SpanBound) -> Self {
        Self { start, end }
    }
}

impl From<(SpanBound, SpanBound)> for Span {
    fn from(value: (SpanBound, SpanBound)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<((usize, usize), (usize, usize))> for Span {
    fn from(value: ((usize, usize), (usize, usize))) -> Self {
        Self::new(value.0.into(), value.1.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    span: Span,
}

struct Cursor<'a> {
    current_line: usize,
    current_col: usize,
    remaining_src: &'a str,
}

impl<'a> Cursor<'a> {
    pub fn new(src: &'a str) -> Self {
        Cursor {
            current_line: 1,
            current_col: 0,
            remaining_src: src,
        }
    }

    pub fn get_next_token(&mut self) -> Option<Token<'a>> {
        if let Some(token) = self.get_ident() {
            return Some(token);
        }
        if let Some(token) = self.get_literal() {
            return Some(token);
        }
        if let Some(token) = self.get_whitespace() {
            return Some(token);
        }
        if let Some(token) = self.get_block_comment() {
            return Some(token);
        }
        if let Some(token) = self.get_line_comment() {
            return Some(token);
        }
        if let Some(token) = self.get_multichar_op() {
            return Some(token);
        }
        if let Some(token) = self.get_simple_token() {
            return Some(token);
        }

        if !self.remaining_src.is_empty() {
            let span = self.consume_src(1);
            Some(Token {
                kind: TokenKind::Invalid(Invalid::UnexpectedChar),
                span,
            })
        } else {
            None
        }
    }

    fn consume_src(&mut self, len: usize) -> Span {
        let src = &self.remaining_src[..len];
        let newlines =
            src.chars().into_iter().fold(
                0usize,
                |count, ch| if ch == '\n' { count + 1 } else { count },
            );

        let start = SpanBound::new(self.current_line, self.current_col);
        let end = if newlines != 0 {
            let last_newline = src.rfind('\n').unwrap();
            let chars_after_last_newline = len - (last_newline + 1);
            SpanBound::new(
                self.current_line + newlines,
                self.current_col + chars_after_last_newline,
            )
        } else {
            SpanBound::new(self.current_line, self.current_col + len)
        };
        self.current_line = end.line;
        self.current_col = end.col;
        self.remaining_src = &self.remaining_src[len..];
        Span::new(start, end)
    }

    fn get_multichar_op(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        let kind = match (src_iter.next()?, src_iter.next()?) {
            ('+', '+') => Some(TokenKind::PostIncrement),
            ('-', '-') => Some(TokenKind::PostDecrement),
            ('<', '=') => Some(TokenKind::LessThanOrEqual),
            ('>', '=') => Some(TokenKind::GreaterThanOrEqual),
            ('=', '=') => Some(TokenKind::Equal),
            ('!', '=') => Some(TokenKind::NotEqual),
            ('&', '&') => Some(TokenKind::LogicalAnd),
            ('|', '|') => Some(TokenKind::LogicalOr),
            _ => None,
        };
        if let Some(kind) = kind {
            let span = self.consume_src(2);
            Some(Token { kind, span })
        } else {
            None
        }
    }

    fn get_simple_token(&mut self) -> Option<Token<'a>> {
        let token_kind = match self.remaining_src.chars().next()? {
            ':' => Some(TokenKind::Colon),
            '=' => Some(TokenKind::AssignmentOp),
            ';' => Some(TokenKind::Semicolon),
            '!' => Some(TokenKind::LogicalNot),
            '^' => Some(TokenKind::Exponentiation),
            '*' => Some(TokenKind::Multiply),
            '/' => Some(TokenKind::Divide),
            '%' => Some(TokenKind::Modulo),
            '+' => Some(TokenKind::Add),
            '-' => Some(TokenKind::Subtract),
            '<' => Some(TokenKind::LessThan),
            '>' => Some(TokenKind::GreaterThan),
            '{' => Some(TokenKind::LeftCurlyBrace),
            '}' => Some(TokenKind::RightCurlyBrace),
            '(' => Some(TokenKind::LeftParen),
            ')' => Some(TokenKind::RightParen),
            '[' => Some(TokenKind::LeftBracket),
            ']' => Some(TokenKind::RightBracket),
            _ => None,
        }?;
        let span = self.consume_src(1);
        Some(Token {
            kind: token_kind,
            span,
        })
    }

    fn get_block_comment(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        if let (Some('/'), Some('*')) = (src_iter.next(), src_iter.next()) {
            if let Some(end_of_block) = self.remaining_src.find("*/") {
                let block_comment = &self.remaining_src[..end_of_block + 2];
                let span = self.consume_src(block_comment.len());
                Some(Token {
                    kind: TokenKind::BlockComment(block_comment),
                    span,
                })
            } else {
                let remaining = &self.remaining_src[..];
                let span = self.consume_src(remaining.len());
                Some(Token {
                    kind: TokenKind::Invalid(Invalid::UnterminatedBlockComment),
                    span,
                })
            }
        } else {
            None
        }
    }

    fn get_line_comment(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        match (src_iter.next()?, src_iter.next()?) {
            ('/', '/') => {
                let src = if let Some(newline) = self.remaining_src.find('\n') {
                    if let Some(cr) = self.remaining_src[..newline].find('\r') {
                        &self.remaining_src[..cr]
                    } else {
                        &self.remaining_src[..newline]
                    }
                } else {
                    &self.remaining_src[..]
                };
                let span = self.consume_src(src.len());
                Some(Token {
                    kind: TokenKind::LineComment(src),
                    span,
                })
            }
            _ => None,
        }
    }

    fn get_whitespace(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        let mut whitespace_len = 0;
        while let Some(ch) = src_iter.next() {
            if ch == '\n' {
                whitespace_len += 1;
            } else if ch.is_ascii_whitespace() {
                whitespace_len += 1;
            } else {
                break;
            }
        }

        if whitespace_len > 0 {
            let span = self.consume_src(whitespace_len);
            Some(Token {
                kind: TokenKind::Whitespace,
                span,
            })
        } else {
            None
        }
    }

    fn get_ident(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        if Self::is_valid_ident_start(src_iter.next()?) {
            let mut ident_len = 1;
            for ch in src_iter {
                if Self::is_valid_ident(ch) {
                    ident_len += 1;
                } else {
                    break;
                }
            }
            let ident_src = &self.remaining_src[..ident_len];
            let ident_span = self.consume_src(ident_len);
            let kind = if let Some(keyword) = convert_ident_to_keyword(ident_src) {
                TokenKind::Keyword(keyword)
            } else {
                TokenKind::Ident(ident_src)
            };
            Some(Token {
                kind,
                span: ident_span,
            })
        } else {
            None
        }
    }

    fn is_valid_ident_start(ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_'
    }

    fn is_valid_ident(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || ch == '_'
    }

    fn get_literal(&mut self) -> Option<Token<'a>> {
        if let Some(token) = self.get_int_literal() {
            return Some(token);
        }
        if let Some(token) = self.get_char_literal() {
            return Some(token);
        }
        if let Some(token) = self.get_string_literal() {
            return Some(token);
        }
        None
    }

    fn get_int_literal(&mut self) -> Option<Token<'a>> {
        // not handling radix yet
        let mut literal_len = 0;
        for ch in self.remaining_src.chars().into_iter() {
            if ch.is_ascii_digit() {
                literal_len += 1;
            } else {
                break;
            }
        }

        if literal_len > 0 {
            let literal_src = &self.remaining_src[..literal_len];
            let literal_span = self.consume_src(literal_len);
            Some(Token {
                kind: TokenKind::IntegerLiteral(literal_src),
                span: literal_span,
            })
        } else {
            None
        }
    }

    fn get_char_literal(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        if src_iter.next()? == '\'' {
            let escaped = match src_iter.next() {
                None | Some('\n' | '\r') => {
                    let span = self.consume_src(1);
                    return Some(Token {
                        kind: TokenKind::Invalid(Invalid::UnterminatedCharLiteral),
                        span,
                    });
                }
                Some('\'') => {
                    let span = self.consume_src(2);
                    return Some(Token {
                        kind: TokenKind::Invalid(Invalid::EmptyCharLiteral),
                        span,
                    });
                }
                Some('\\') => true,
                Some(_) => false,
            };
            let mut terminated = false;
            let mut literal_len = 2;
            while let Some(ch) = src_iter.next() {
                literal_len += 1;
                if ch == '\'' {
                    terminated = true;
                    break;
                }
            }
            let literal_src = &self.remaining_src[..literal_len];
            let span = self.consume_src(literal_len);
            if terminated {
                if literal_len == 3 && !escaped {
                    Some(Token {
                        kind: TokenKind::CharLiteral(literal_src),
                        span,
                    })
                } else if literal_len == 4 && escaped {
                    Some(Token {
                        kind: TokenKind::CharLiteral(literal_src),
                        span,
                    })
                } else {
                    Some(Token {
                        kind: TokenKind::Invalid(Invalid::CharLiteralTooLong),
                        span,
                    })
                }
            } else {
                Some(Token {
                    kind: TokenKind::Invalid(Invalid::UnterminatedCharLiteral),
                    span,
                })
            }
        } else {
            None
        }
    }

    fn get_string_literal(&mut self) -> Option<Token<'a>> {
        let mut src_iter = self.remaining_src.chars().into_iter();
        if src_iter.next()? == '"' {
            let mut string_len = 1;
            let mut terminated = false;
            while let Some(ch) = src_iter.next() {
                if ch == '\n' || ch == '\r' {
                    break;
                }
                string_len += 1;
                if ch == '\\' {
                    match src_iter.next() {
                        Some(_) => string_len += 1,
                        None => break,
                    }
                } else if ch == '"' {
                    terminated = true;
                    break;
                }
            }
            let literal = &self.remaining_src[..string_len];
            let span = self.consume_src(string_len);
            if terminated {
                Some(Token {
                    kind: TokenKind::StringLiteral(literal),
                    span,
                })
            } else {
                Some(Token {
                    kind: TokenKind::Invalid(Invalid::UnterminatedStringLiteral),
                    span,
                })
            }
        } else {
            None
        }
    }
}

pub fn scan(source: &str) -> Vec<Token> {
    let mut tokens = vec![];

    let mut cursor = Cursor::new(source);
    while let Some(token) = cursor.get_next_token() {
        tokens.push(token);
    }

    tokens
}

fn convert_ident_to_keyword(ident: &str) -> Option<Keyword> {
    match Keyword::from_str(ident) {
        Ok(key) => Some(key),
        Err(()) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_token_kinds_minus_whitespace(expected_kinds: &[TokenKind], tokens: &[Token]) {
        let tokens = tokens
            .into_iter()
            .filter(|token| !matches!(token.kind, TokenKind::Whitespace))
            .collect::<Vec<_>>();

        expected_kinds.into_iter().zip(tokens.into_iter()).for_each(
            |(expected_kind, actual_token)| {
                assert_eq!(*expected_kind, actual_token.kind);
            },
        );
    }

    fn assert_has_invalid_tokens(tokens: &[Token]) {
        assert!(tokens
            .into_iter()
            .find(|token| matches!(token.kind, TokenKind::Invalid(_)))
            .is_some())
    }

    #[test]
    fn test_integer_initialization() {
        let src = "x: integer = 65;";
        let tokens = scan(src);
        let expected_tokens: Vec<Token> = vec![
            Token {
                kind: TokenKind::Ident("x"),
                span: Span::from(((1, 0), (1, 1))),
            },
            Token {
                kind: TokenKind::Colon,
                span: Span::from(((1, 1), (1, 2))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 2), (1, 3))),
            },
            Token {
                kind: TokenKind::Keyword(Keyword::Integer),
                span: Span::from(((1, 3), (1, 10))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 10), (1, 11))),
            },
            Token {
                kind: TokenKind::AssignmentOp,
                span: Span::from(((1, 11), (1, 12))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 12), (1, 13))),
            },
            Token {
                kind: TokenKind::IntegerLiteral("65"),
                span: Span::from(((1, 13), (1, 15))),
            },
            Token {
                kind: TokenKind::Semicolon,
                span: Span::from(((1, 15), (1, 16))),
            },
        ];

        assert_eq!(expected_tokens, tokens);
    }

    #[test]
    fn test_char_initialization() {
        let src = "y: char = 'A';";
        let tokens = scan(src);
        let expected_tokens: Vec<Token> = vec![
            Token {
                kind: TokenKind::Ident("y"),
                span: Span::from(((1, 0), (1, 1))),
            },
            Token {
                kind: TokenKind::Colon,
                span: Span::from(((1, 1), (1, 2))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 2), (1, 3))),
            },
            Token {
                kind: TokenKind::Keyword(Keyword::Char),
                span: Span::from(((1, 3), (1, 7))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 7), (1, 8))),
            },
            Token {
                kind: TokenKind::AssignmentOp,
                span: Span::from(((1, 8), (1, 9))),
            },
            Token {
                kind: TokenKind::Whitespace,
                span: Span::from(((1, 9), (1, 10))),
            },
            Token {
                kind: TokenKind::CharLiteral("'A'"),
                span: Span::from(((1, 10), (1, 13))),
            },
            Token {
                kind: TokenKind::Semicolon,
                span: Span::from(((1, 13), (1, 14))),
            },
        ];

        assert_eq!(expected_tokens, tokens);
    }

    #[test]
    fn test_cacophony_of_exclamation_and_equality() {
        let src = "!! !! !===!!==!==!!==!=  &&";
        let tokens = scan(src);

        let expected_kinds = vec![
            TokenKind::LogicalNot,
            TokenKind::LogicalNot,
            TokenKind::LogicalNot,
            TokenKind::LogicalNot,
            TokenKind::NotEqual,
            TokenKind::Equal,
            TokenKind::LogicalNot,
            TokenKind::NotEqual,
            TokenKind::AssignmentOp,
            TokenKind::NotEqual,
            TokenKind::AssignmentOp,
            TokenKind::LogicalNot,
            TokenKind::NotEqual,
            TokenKind::AssignmentOp,
            TokenKind::NotEqual,
            TokenKind::LogicalAnd,
        ];

        assert_token_kinds_minus_whitespace(&expected_kinds, &tokens);
    }

    #[test]
    fn test_string_madness() {
        let src = r#"" \'\"\'\"\' ""#;
        let tokens = scan(src);

        let expected_kinds = vec![TokenKind::StringLiteral(r#"" \'\"\'\"\' ""#)];

        assert_token_kinds_minus_whitespace(&expected_kinds, &tokens);
    }

    #[test]
    fn test_simple_string_with_escape() {
        let src = r#""\'""#;
        let tokens = scan(src);

        let expected_kinds = vec![TokenKind::StringLiteral(r#""\'""#)];
        assert_token_kinds_minus_whitespace(&expected_kinds, &tokens);
    }

    #[test]
    fn test_simple_char() {
        let src = r#"'A'"#;
        let tokens = scan(src);

        let expected_kinds = vec![TokenKind::CharLiteral(r#"'A'"#)];
        assert_token_kinds_minus_whitespace(&expected_kinds, &tokens)
    }

    #[test]
    fn test_newlines_chars() {
        let valid_src = "'\\n'";
        let tokens = scan(valid_src);
        assert_token_kinds_minus_whitespace(&[TokenKind::CharLiteral(r#"'\n'"#)], &tokens);

        let invalid_src = "'\n'";
        let tokens = scan(invalid_src);
        assert_has_invalid_tokens(&tokens);
    }

    #[test]
    fn test_int_literal_with_line_comment() {
        let src = "65 // explain the number";
        let tokens = scan(src);

        assert_token_kinds_minus_whitespace(
            &[
                TokenKind::IntegerLiteral("65"),
                TokenKind::LineComment("// explain the number"),
            ],
            &tokens,
        );
    }

    #[test]
    fn test_int_literal_with_simple_block_comment() {
        let src = r#"65 /*
            comment about num
            */"#;
        let tokens = scan(src);

        assert_token_kinds_minus_whitespace(
            &[
                TokenKind::IntegerLiteral("65"),
                TokenKind::BlockComment("/*\n            comment about num\n            */"),
            ],
            &tokens,
        )
    }
}
