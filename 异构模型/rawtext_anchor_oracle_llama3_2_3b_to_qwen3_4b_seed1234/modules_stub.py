import torch.nn.functional as F


def full_kl(student, teacher, temperature=1.0):
    student_log = F.log_softmax(student.float() / temperature, -1)
    teacher_log = F.log_softmax(teacher.detach().float() / temperature, -1)
    return F.kl_div(student_log, teacher_log, reduction="sum", log_target=True) * temperature**2
