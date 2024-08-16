# %%
problem_text = 'Provide me with the complete, valid problem PDDL file that \
    describes the following planning problem directly without further \
    explanations or texts.\n\n'
domain_text = 'The domain for the planning problem is:\n\n\
    ```\n\
    {domain}\n\
    ```'
elicitation_text = 'The natural language description of the problem is:\n\n\
    ```\n\
    {natural_language_description}\n\
    ```\n\n\
    Write the PDDL code for the problem here:\n\n\
    ```\n'

DOMAINS = {
    'blocksworld': """;; source: https://github.com/AI-Planning/pddl-generators/blob/main/blocksworld/domain.pddl
    ;; same as used in IPC 2023
    ;;
    (define (domain blocksworld)

    (:requirements :strips)

    (:predicates (clear ?x)
                (on-table ?x)
                (arm-empty)
                (holding ?x)
                (on ?x ?y))

    (:action pickup
    :parameters (?ob)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
                (not (arm-empty))))

    (:action putdown
    :parameters  (?ob)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob)
                (not (holding ?ob))))

    (:action stack
    :parameters  (?ob ?underob)
    :precondition (and (clear ?underob) (holding ?ob))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
                (not (clear ?underob)) (not (holding ?ob))))

    (:action unstack
    :parameters  (?ob ?underob)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) (clear ?underob)
                (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
    """,
    'gripper': """;; source: https://github.com/AI-Planning/pddl-generators/blob/main/gripper/domain.pddl
    (define (domain gripper)
       (:requirements :strips)
       (:predicates (room ?r)
            (ball ?b)
            (gripper ?g)
            (at-robby ?r)
            (at ?b ?r)
            (free ?g)
            (carry ?o ?g))

       (:action move
           :parameters  (?from ?to)
           :precondition (and  (room ?from) (room ?to) (at-robby ?from))
           :effect (and  (at-robby ?to)
                 (not (at-robby ?from))))

       (:action pick
           :parameters (?obj ?room ?gripper)
           :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                    (at ?obj ?room) (at-robby ?room) (free ?gripper))
           :effect (and (carry ?obj ?gripper)
                (not (at ?obj ?room))
                (not (free ?gripper))))

       (:action drop
           :parameters  (?obj  ?room ?gripper)
           :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
                    (carry ?obj ?gripper) (at-robby ?room))
           :effect (and (at ?obj ?room)
                (free ?gripper)
                (not (carry ?obj ?gripper)))))
    """,
}


def make_prompt(natural_language_description: str, domain_name: str) -> str:
    return f'{problem_text}{domain_text.format(domain=DOMAINS[domain_name])}{elicitation_text.format(natural_language_description=natural_language_description)}'
