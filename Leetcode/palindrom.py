def main():
    def isPalindrome(x):
        """
        :type x: int
        :rtype: bool
        """
        x = str(x)
        if len(x) % 2 == 0:
            if len(x) < 0:
                a = 'Не палиндром'
            s1 = x[:len(x) // 2]
            s2 = x[len(x) // 2:]
            s2 = s2[::-1]
            if s1 == s2:
                a = 'Палиндром'
            else:
                a = 'Не палиндром'
        else:
            s1 = x[:len(x) // 2]
            s2 = x[len(x) // 2 + 1:]
            if s1 == s2:
                a = 'Палиндром'
            else:
                a = 'Не палиндром'
        print(a)

    isPalindrome(123321)


if __name__ == "__main__":
    main()
