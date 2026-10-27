(define (stream masstransport3d)
  (:stream sample-grasp
    :inputs (?o)
    :domain (Item ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream sample-carry-conf
    :inputs (?g)
    :domain (ToteGrasp ?g)
    :outputs (?a)
    :certified (CarryConf ?g ?a)
  )
  ; Item spots and the bases that serve them are fixed, so the arm paths for a spot
  ; are planned once and shared by every skeleton that uses it.
  (:stream inverse-kinematics
    :inputs (?o ?p ?g ?r ?q)
    :domain (and (Pose ?o ?p) (PoseIn ?p ?r) (Grasp ?o ?g) (TableBase ?q ?r))
    :outputs (?t)
    :certified (and (ATraj ?t) (Kin ?o ?p ?g ?q ?t))
  )
  (:stream slot-inverse-kinematics
    :inputs (?o ?s ?tp ?g ?r ?q)
    :domain (and (Item ?o) (Slot ?s) (TotePose ?tp) (TotePoseIn ?tp ?r)
                 (Grasp ?o ?g) (TableBase ?q ?r))
    :outputs (?t)
    :certified (and (ATraj ?t) (SlotKin ?o ?s ?tp ?g ?q ?t))
  )
  ; The carry conf fixes where the base must stand for a tote pose. Its region is
  ; an input, so even an optimistic base conf is on one side only.
  (:stream tote-inverse-kinematics
    :inputs (?tp ?r ?g ?a)
    :domain (and (TotePose ?tp) (TotePoseIn ?tp ?r) (CarryConf ?g ?a))
    :outputs (?q ?t)
    :certified (and (BConf ?q) (BConfIn ?q ?r) (ATraj ?t) (ToteKin ?tp ?g ?a ?q ?t))
  )
  (:stream plan-base-motion
    :inputs (?q1 ?q2)
    :domain (and (BConf ?q1) (BConf ?q2))
    :outputs (?t)
    :certified (BaseMotion ?q1 ?t ?q2)
  )

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2)
  )
  (:stream test-cfree-pose-tote
    :inputs (?o ?p ?tp)
    :domain (and (Pose ?o ?p) (TotePose ?tp))
    :certified (CFreePoseTote ?o ?p ?tp)
  )
  (:stream test-cfree-traj-pose
    :inputs (?t ?o2 ?p2)
    :domain (and (ATraj ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?t ?o2 ?p2)
  )
  (:stream test-cfree-traj-tote
    :inputs (?t ?tp)
    :domain (and (ATraj ?t) (TotePose ?tp))
    :certified (CFreeTrajTote ?t ?tp)
  )
  ; Whether the arm can hold the tote with whatever is in its slots, at both ends
  ; of the vertical lift and along it. This is where the item masses enter.
  (:stream test-carry-torque
    :inputs (?g ?a)
    :domain (CarryConf ?g ?a)
    :fluents (InSlot)
    :certified (CarryOK ?g ?a)
  )
)
