extern crate typed_arena;
extern crate smallvec;
extern crate ref_eq;

use std::collections::HashMap;
use std::fmt;
use std::cell::RefCell;
use ref_eq::ref_eq;

mod multirefmap;

#[derive(Copy, Clone, Debug)]
enum BinOp {
    Add, Sub, Mul, Div
}

#[derive(Copy, Clone, Debug)]
enum UnOp {
    Negate
}

type ExprRef<'a> = &'a Expr<'a>;

trait Visitor<'a, 't: 'a> {
    fn factory(&self) -> &'a NodeFactory<'a, 't>;

    fn visit_func(&self, func: FuncRef<'a>) -> FuncRef<'a> {
        unimplemented!()
    }

    fn visit_expr(&self, node: ExprRef<'a>) -> ExprRef<'a> {
        self.walk_expr(node)
    }
    
    fn walk_expr(&self, node: ExprRef<'a>) -> ExprRef<'a> {
        match &node.data {
            &ExprData::Binary(op, l, r) => {
                let vl = self.visit_expr(l);
                let vr = self.visit_expr(r);
                if ref_eq(l, vl) && ref_eq(r, vr) {
                    node
                } else {
                    self.factory().bin(op, vl, vr)
                }
            }
            &ExprData::Unary(op, n) => {
                let vn = self.visit_expr(n);
                if ref_eq(n, vn) {
                    node
                } else {
                    self.factory().unary(op, vn)
                }
            }
            &ExprData::Call(func, ref args) => {
                let visited_func = self.visit_func(func);
                let visited_args = self.visit_exprs(args);
                unimplemented!()
            }
            &ExprData::Parameter(parameter) => {
                node
            }
            &ExprData::Integer(_) => {
                node
            }
        }
    }

    fn visit_parameter(&self, parameter: &'a Parameter<'a>) -> &'a Parameter<'a> {
        unimplemented!()
    }

    fn visit_exprs(&self, exprs: &'a Vec<ExprRef<'a>>) -> &'a Vec<ExprRef<'a>> {
        unimplemented!()
    }
}

struct Expr<'a> {
    data: ExprData<'a>,
    ty: Option<TypeRef<'a>>
}

impl<'a> std::fmt::Debug for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.data)?;
        if let Some(ty) = self.ty {
            write!(f, ":{:?}", ty)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Parameter<'a> {
    name: String,
    ty: TypeRef<'a>
}

#[derive(Debug)]
struct Func<'a> {
    name: String,
    body: ExprRef<'a>,
    parameters: Vec<Parameter<'a>>
}

type FuncRef<'a> = &'a Func<'a>;

#[derive(Debug)]
enum ExprData<'a> {
    Integer(u64),
    Binary(BinOp, ExprRef<'a>, ExprRef<'a>),
    Unary(UnOp, ExprRef<'a>),
    Parameter(usize),
    Call(&'a Func<'a>, Vec<ExprRef<'a>>)
}

type TypeRef<'t> = &'t Type<'t>;
// smallvec is not working
//type TypeVec<'t> = smallvec::SmallVec<[TypeRef<'t>; 4]>;
type TypeVec<'t> = Vec<TypeRef<'t>>;

// #[derive(Debug)]
// enum TypeVec<'t> {
//     Zero,
//     One(TypeRef<'t>),
//     Two(TypeRef<'t>),
//     More(Vec<TypeRef<'t>>)
// }

#[derive(Debug)]
enum Type<'t> {
    Named(String),
    GenericInstance(TypeRef<'t>, TypeVec<'t>)
}


struct TypeContext<'t> {
    types: typed_arena::Arena<Type<'t>>,
    map: RefCell<HashMap<&'t str, TypeRef<'t>>>
}

impl<'t> TypeContext<'t> {
    pub fn new() -> TypeContext<'t> {
        TypeContext { types: typed_arena::Arena::new(), map: RefCell::new(HashMap::new()) }
    }

    pub fn named(&'t self, name: &str) -> TypeRef<'t> {
        if let Some(entry) = self.map.borrow().get(name) {
            return entry;
        }

        let name = name.to_string();
        let ty: &Type = self.types.alloc(Type::Named(name));
        if let &Type::Named(ref s) = ty {
            self.map.borrow_mut().insert(s.as_str(), ty);
        }
        ty
    }
}

struct NodeFactory<'a, 't: 'a> {
    nodes: typed_arena::Arena<Expr<'a>>,
    tc: &'t TypeContext<'t>
}

impl<'a, 't> NodeFactory<'a, 't> {
    pub fn new(tc: &'t TypeContext<'t>) -> NodeFactory<'a, 't> {
        NodeFactory {
            nodes: typed_arena::Arena::new(),
            tc
        }
    }

    pub fn unary(&'a self, op: UnOp, node: ExprRef<'a>) -> ExprRef<'a> {
        self.node(ExprData::Unary(op, node), None)
    }

    pub fn integer(&'a self, value: u64) -> ExprRef<'a> {
        self.node(ExprData::Integer(value), None)
    }

    pub fn bin(&'a self, op: BinOp, left: ExprRef<'a>, right: ExprRef<'a>) -> ExprRef<'a> {
        self.node(ExprData::Binary(op, left, right), None)
    }

    fn node(&'a self, data: ExprData<'a>, ty: Option<TypeRef<'a>>) -> ExprRef<'a> {
        self.nodes.alloc(Expr { data, ty })
    }
}

struct NegateVisitor<'a, 't: 'a> {
    nf: &'a NodeFactory<'a, 't>
}

impl<'a,'t> Visitor<'a,'t> for NegateVisitor<'a,'t> {
    fn factory(&self) -> &'a NodeFactory<'a, 't> {
        self.nf
    }

    fn visit_expr(&self, node: ExprRef<'a>) -> ExprRef<'a> {
        let node = self.walk_expr(node);
        match &node.data {
            &ExprData::Binary(BinOp::Add, ref l, ref r) => {
                self.factory().bin(BinOp::Sub, l, r)
            }
            &ExprData::Integer(value) => {
                self.factory().integer(value + 10)
            }
            _ => node
        }
    }
}

struct InlineVisitor<'a, 't: 'a> {
    nf: &'a NodeFactory<'a, 't>,
    visited: multirefmap::RefCache<'a>
}

impl<'a,'t> Visitor<'a,'t> for InlineVisitor<'a,'t> {
    fn factory(&self) -> &'a NodeFactory<'a, 't> {
        self.nf
    }

    fn visit_expr(&self, node: ExprRef<'a>) -> ExprRef<'a> {
        self.visited.get_or_insert(node, ||  {
            let node = self.walk_expr(node);
            match &node.data {
                &ExprData::Binary(BinOp::Add, ref l, ref r) => {
                    self.factory().bin(BinOp::Sub, l, r)
                }
                &ExprData::Integer(value) => {
                    self.factory().integer(value + 10)
                }
                _ => node
            }
        })
    }
}

fn main() {
    let tc = TypeContext::new();
    {
        let f = NodeFactory::new(&tc);
        let l = f.integer(1);
        let r = f.integer(2);
        let added = f.bin(BinOp::Add, f.integer(1), f.unary(UnOp::Negate, f.integer(222)));
        println!("{:?}", added);
        let mut visitor = NegateVisitor { nf: &f };
        let inversed = visitor.visit_expr(added);
        println!("{:?}", inversed);
    }
}
